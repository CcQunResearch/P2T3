# P2T3

Source code for P2T3 in paper: 

**Addressing Over-Smoothing in Social Media Rumor Detection with Pre-Trained Propagation Tree Transformer**

## Run

The two training strategies in the paper can be run in the following ways:

```shell script
nohup python main\(pretrain\).py &
nohup python main\(finetuning\).py &
```

## Dependencies

- [pytorch](https://pytorch.org/) == 1.12.0

- [transformers](https://github.com/huggingface/transformers) == 4.2.1

- [gensim](https://radimrehurek.com/gensim/index.html) == 4.0.1