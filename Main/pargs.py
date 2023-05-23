import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    # dataset hyperparameters
    parser.add_argument('--dataset', type=str, default='Weibo')  # fine-tune dataset
    parser.add_argument('--unsup_dataset', type=str, default='UWeibo')  # pretrain dataset

    # word embedding hyperparameters
    parser.add_argument('--embedding_method', type=str, default='word2vec')
    parser.add_argument('--word_embedding_size', type=int, default=512)
    parser.add_argument('--cn_word_tokenization', type=str2bool,
                        default=False)  # character or word, only for Chinese dataset
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabeled data train size', default=300000)

    # pretrain and finetune
    # embedding hyperparameters
    parser.add_argument('--use_chain_identifier', type=str2bool, default=True)
    parser.add_argument('--use_depth_embedding', type=str2bool, default=True)
    parser.add_argument('--use_type_embedding', type=str2bool, default=True)
    parser.add_argument('--max_sequence_length', type=int, default=1000)

    # pretrain and finetune
    # model hyperparameters
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--measure', type=str, default='JSD')  # GAN, JSD, X2, KL, RKL, DV, H2, W1
    parser.add_argument('--pooling', type=str, default='level1')  # level1 or mean

    # only pretrain
    # pretrain hyperparameters
    parser.add_argument('--pt_batch_size', type=int, default=8)
    parser.add_argument('--pt_acc_batch_size', type=int, default=800)
    parser.add_argument('--pt_num_epochs', type=int, default=30)
    parser.add_argument('--pt_lr', type=float, default=5e-5)
    parser.add_argument('--pt_weight_decay', type=float, default=1e-3)
    parser.add_argument('--pt_warmup_ratio', type=float, default=0.1)

    # only finetune
    # finetune hyperparameters
    parser.add_argument('--ft_runs', type=int, default=5)
    parser.add_argument('--ft_batch_size', type=int, default=4)
    parser.add_argument('--ft_acc_batch_size', type=int, default=16)
    parser.add_argument('--ft_num_epochs', type=int, default=120)
    parser.add_argument('--ft_lr', type=float, default=5e-5)
    parser.add_argument('--ft_weight_decay', type=float, default=1e-3)
    parser.add_argument('--ft_warmup_ratio', type=float, default=0.1)
    parser.add_argument('--split', type=str, default='802')
    parser.add_argument('--lamda', type=float, default=0.001)

    # cuda hyperparameters
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default="0,1,2,3")  # parallel

    args = parser.parse_args()
    return args
