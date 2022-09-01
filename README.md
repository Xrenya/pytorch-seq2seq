# PyTorch Seq2Seq

The main difference from original repo is updated torchtext==0.12.0 with some additional functions.  

## Tutorials

* 1. - [Sequence to Sequence Learning with Neural Networks](https://github.com/Xrenya/pytorch-seq2seq/blob/main/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xrenya/pytorch-seq2seq/blob/main/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb)

    This first tutorial covers the workflow of a PyTorch with torchtext seq2seq project. We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these models in PyTorch, and how to use torchtext to do all of the heavy lifting with regards to text processing. The model itself will be based off an implementation of Sequence to Sequence Learning with Neural Networks, which uses multi-layer LSTMs.

* 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/Xrenya/pytorch-seq2seq/blob/main/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_ipynb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xrenya/pytorch-seq2seq/blob/main/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_ipynb.ipynb)
    
    Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch and torchtext gained from the previous tutorial, we'll cover a second second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

## Reference
1. https://github.com/bentrevett/pytorch-seq2seq
2. https://github.com/spro/practical-pytorch
3. https://github.com/keon/seq2seq
4. https://github.com/pengshuang/CNN-Seq2Seq
5. https://github.com/pytorch/fairseq
6. https://github.com/jadore801120/attention-is-all-you-need-pytorch
7. http://nlp.seas.harvard.edu/2018/04/03/attention.html
8. https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
