# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 16:43
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main(finetuning).py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Main.pargs import pargs
from Main.utils import create_log_dict_finetuning, write_json, write_log
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.dataset import SequenceTreeDataset
from Main.model import P2T3
from Main.collator import Collator
from Main.sort import sort_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import get_linear_schedule_with_warmup, AdamW


def fine_tuning(train_loader, model, optimizer, scheduler, lamda):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        padded_sequences, attention_mask, indexes, y = batch
        pred = model(padded_sequences, attention_mask)
        sup_loss = F.nll_loss(pred, y.long().view(-1))
        unsup_loss = model.unsup_loss(padded_sequences, attention_mask, indexes)

        loss = sup_loss + lamda * unsup_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * padded_sequences.size(0)

    return total_loss / len(train_loader.dataset)


def test(model, dataloader, num_classes):
    model.eval()
    error = 0

    y_true = []
    y_pred = []

    for batch in dataloader:
        padded_sequences, attention_mask, indexes, y = batch
        pred = model(padded_sequences, attention_mask)
        error += F.nll_loss(pred, y.long().view(-1)).item() * padded_sequences.size(0)
        y_true += y.tolist()
        y_pred += pred.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=True), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=True), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=True), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / len(dataloader.dataset), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]


def test_and_log(model, val_loader, test_loader, num_classes, epoch, lr, loss, train_acc, ft_log_record):
    val_error, val_acc, val_precs, val_recs, val_f1s, val_micro_metric, val_macro_metric = \
        test(model, val_loader, num_classes)
    test_error, test_acc, test_precs, test_recs, test_f1s, test_micro_metric, test_macro_metric = \
        test(model, test_loader, num_classes)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}'

    ft_log_record['val accs'].append(val_acc)
    ft_log_record['test accs'].append(test_acc)
    ft_log_record['test precs'].append(test_precs)
    ft_log_record['test recs'].append(test_recs)
    ft_log_record['test f1s'].append(test_f1s)
    ft_log_record['test micro metric'].append(test_micro_metric)
    ft_log_record['test macro metric'].append(test_macro_metric)
    return val_error, log_info, ft_log_record


if __name__ == '__main__':
    args = pargs()

    cn_datasets = ['DRWeibo', 'Weibo', 'UWeibo']
    en_datasets = ['PHEME9', 'Twitter', 'UTwitter']

    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    num_classes = 4 if 'Twitter' in dataset or dataset == 'PHEME' else 2

    unsup_train_size = args.unsup_train_size
    embedding_method = args.embedding_method
    word_embedding_size = args.word_embedding_size
    cn_word_tokenization = args.cn_word_tokenization

    lang, tokenize_mode, content_key, word2vec_datasets = 'en', 'word', 'content', en_datasets
    if unsup_dataset == 'UWeibo':
        lang = 'cn'
        word2vec_datasets = cn_datasets
        if cn_word_tokenization:
            content_key = 'word content'
        else:
            tokenize_mode = 'char'
    word2vec_model_path = osp.join(dirname, '..', 'Model',
                                   f'w2v_{lang}_{tokenize_mode}_{unsup_train_size}_{word_embedding_size}.model')

    use_chain_identifier = args.use_chain_identifier
    use_depth_embedding = args.use_depth_embedding
    use_type_embedding = args.use_type_embedding
    max_sequence_length = args.max_sequence_length

    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    dim_feedforward = args.dim_feedforward
    measure = args.measure
    pooling = args.pooling

    pt_batch_size = args.pt_batch_size
    pt_acc_batch_size = args.pt_acc_batch_size
    pt_num_epochs = args.pt_num_epochs
    pt_lr = args.pt_lr
    pt_weight_decay = args.pt_weight_decay
    pt_warmup_ratio = args.pt_warmup_ratio

    device = args.gpu if args.cuda else 'cpu'
    ft_runs = args.ft_runs
    ft_batch_size = args.ft_batch_size
    ft_num_epochs = args.ft_num_epochs
    ft_weight_decay = args.ft_weight_decay
    ft_warmup_ratio = args.ft_warmup_ratio
    split = args.split
    lamda = args.lamda

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'[F]{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'[F]{log_name}.json')

    model_unique_hyparams = [unsup_dataset, tokenize_mode, unsup_train_size, word_embedding_size, d_model, num_layers,
                             num_heads, dim_feedforward, measure, pooling, pt_acc_batch_size, pt_num_epochs, pt_lr,
                             pt_weight_decay, pt_warmup_ratio]
    prior_setting = "".join(
        ["C" if use_chain_identifier else "", "D" if use_depth_embedding else "", "T" if use_type_embedding else ""])
    saved_model_name = f'{"_".join([str(hyparam) for hyparam in model_unique_hyparams])}_{prior_setting}.pt'
    weight_path = osp.join(dirname, '..', 'Model', saved_model_name)

    log = open(log_path, 'w')
    write_log(log, f'Fine Tuning')
    log_dict = create_log_dict_finetuning(args)
    log_dict['pretrained model name'] = saved_model_name
    write_json(log_dict, log_json_path)

    if not osp.exists(word2vec_model_path) and embedding_method == 'word2vec':
        sentences = collect_sentences(word2vec_datasets, unsup_train_size, content_key)
        w2v_model = train_word2vec(sentences, word_embedding_size)
        w2v_model.save(word2vec_model_path)

    word2vec = Embedding(word2vec_model_path) if embedding_method == 'word2vec' else None
    bert_model = None

    # ks = [10, 20, 40, 80, 100, 200, 300, 500, 10000]
    ks = [10000]
    for k in ks:
        for r in range(ft_runs):
            ft_lr = args.ft_lr
            write_log(log, f'k:{k}, r:{r}')

            ft_log_record = {'k': k, 'r': r, 'val accs': [], 'test accs': [], 'test precs': [], 'test recs': [],
                             'test f1s': [], 'test micro metric': [], 'test macro metric': []}
            sort_dataset(label_source_path, label_dataset_path, k_shot=k, split=split)

            train_dataset = SequenceTreeDataset(train_path, d_model, embedding_method, word2vec, bert_model,
                                                content_key, use_chain_identifier, use_depth_embedding,
                                                use_type_embedding, max_sequence_length)
            val_dataset = SequenceTreeDataset(val_path, d_model, embedding_method, word2vec, bert_model,
                                              content_key, use_chain_identifier, use_depth_embedding,
                                              use_type_embedding, max_sequence_length)
            test_dataset = SequenceTreeDataset(test_path, d_model, embedding_method, word2vec, bert_model,
                                               content_key, use_chain_identifier, use_depth_embedding,
                                               use_type_embedding, max_sequence_length)
            collator = Collator(pooling, device, output_label=True)
            train_loader = DataLoader(train_dataset, ft_batch_size, collate_fn=collator, shuffle=True)
            val_loader = DataLoader(val_dataset, ft_batch_size, collate_fn=collator, shuffle=True)
            test_loader = DataLoader(test_dataset, ft_batch_size, collate_fn=collator, shuffle=True)

            model = P2T3(num_layers, d_model, num_heads, dim_feedforward, word_embedding_size, pooling, measure).to(
                device)
            # model.load_state_dict(torch.load(weight_path))
            model.lin_class = Linear(d_model, num_classes).to(device)

            # optimizer = Adam(model.parameters(), lr=ft_lr, weight_decay=ft_weight_decay)
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

            total_steps = len(train_loader) * ft_num_epochs
            warmup_steps = int(ft_warmup_ratio * total_steps)
            optimizer = AdamW(model.parameters(), lr=ft_lr, weight_decay=ft_weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)

            val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes, 0,
                                                              args.ft_lr, 0, 0, ft_log_record)
            write_log(log, log_info)

            for epoch in range(1, ft_num_epochs + 1):
                ft_lr = scheduler.optimizer.param_groups[0]['lr']
                _ = fine_tuning(train_loader, model, optimizer, scheduler, lamda)

                train_error, train_acc, _, _, _, _, _ = test(model, train_loader, num_classes)
                val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes, epoch,
                                                                  ft_lr, train_error, train_acc, ft_log_record)
                write_log(log, log_info)

                # if split == '622':
                #     scheduler.step(val_error)

            ft_log_record['mean acc'] = round(np.mean(ft_log_record['test accs'][-10:]), 3)
            log_dict['record'].append(ft_log_record)
            write_log(log, '')
            write_json(log_dict, log_json_path)
