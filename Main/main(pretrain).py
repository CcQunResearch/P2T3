# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 20:07
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main(pretrain).py
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
import torch
from torch.utils.data import DataLoader
from Main.pargs import pargs
from Main.utils import create_log_dict_pretrain, write_json, write_log
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.dataset import SequenceTreeDataset
from Main.model import P2T3
from Main.collator import Collator
from transformers import AdamW, get_linear_schedule_with_warmup

if __name__ == '__main__':
    args = pargs()

    # The last one in the list is the unsupervised dataset
    cn_datasets = ['DRWeibo', 'Weibo', 'UWeibo']
    en_datasets = ['PHEME9', 'Twitter', 'UTwitter']

    unsup_dataset = args.unsup_dataset
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

    device = args.gpu if args.cuda else 'cpu'
    pt_batch_size = args.pt_batch_size
    pt_acc_batch_size = args.pt_acc_batch_size
    pt_num_epochs = args.pt_num_epochs
    pt_lr = args.pt_lr
    pt_weight_decay = args.pt_weight_decay
    pt_warmup_ratio = args.pt_warmup_ratio

    unlabeled_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'[P]{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'[P]{log_name}.json')

    model_unique_hyparams = [unsup_dataset, tokenize_mode, unsup_train_size, word_embedding_size, d_model, num_layers,
                             num_heads, dim_feedforward, measure, pooling, pt_acc_batch_size, pt_num_epochs, pt_lr,
                             pt_weight_decay, pt_warmup_ratio]
    prior_setting = "".join(
        ["C" if use_chain_identifier else "", "D" if use_depth_embedding else "", "T" if use_type_embedding else ""])
    saved_model_name = f'{"_".join([str(hyparam) for hyparam in model_unique_hyparams])}_{prior_setting}.pt'
    weight_path = osp.join(dirname, '..', 'Model', saved_model_name)

    log = open(log_path, 'w')
    write_log(log, f'Pretraining')
    log_dict = create_log_dict_pretrain(args)
    log_dict['saved model name'] = saved_model_name
    write_json(log_dict, log_json_path)

    if not osp.exists(word2vec_model_path) and embedding_method == 'word2vec':
        sentences = collect_sentences(word2vec_datasets, unsup_train_size, content_key)
        w2v_model = train_word2vec(sentences, word_embedding_size)
        w2v_model.save(word2vec_model_path)

    word2vec = Embedding(word2vec_model_path) if embedding_method == 'word2vec' else None
    bert_model = None

    # data
    unlabeled_dataset = SequenceTreeDataset(unlabeled_dataset_path, d_model, embedding_method, word2vec, bert_model,
                                            content_key, use_chain_identifier, use_depth_embedding, use_type_embedding,
                                            max_sequence_length)
    collator = Collator(pooling, device)
    unsup_train_loader = DataLoader(unlabeled_dataset, pt_batch_size, collate_fn=collator, shuffle=True)

    # model
    model = P2T3(num_layers, d_model, num_heads, dim_feedforward, word_embedding_size, pooling, measure).to(device)

    # optimizer
    grad_accumulation_steps = pt_acc_batch_size // pt_batch_size
    total_steps = len(unsup_train_loader) * pt_num_epochs // grad_accumulation_steps
    warmup_steps = int(pt_warmup_ratio * total_steps)
    optimizer = AdamW(model.parameters(), lr=pt_lr, weight_decay=pt_weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    step = 0
    model.train()
    grad_accumulation_counter = 0
    for epoch in range(pt_num_epochs):
        for batch in unsup_train_loader:
            padded_sequences, attention_mask, indexes = batch
            loss = model.unsup_loss(padded_sequences, attention_mask, indexes)
            loss = loss / grad_accumulation_steps
            loss.backward()

            grad_accumulation_counter += 1
            if grad_accumulation_counter == grad_accumulation_steps:
                optimizer.step()
                scheduler.step()  # update learning rate schedule
                optimizer.zero_grad()
                grad_accumulation_counter = 0
                step += 1

        write_log(log, f"Epoch: {epoch + 1}, Loss: {loss.item()}, Training Step: {step}")
    torch.save(model.state_dict(), weight_path)
