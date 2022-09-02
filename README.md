# PyTorch Seq2Seq

The main difference from original repo is updated torchtext>=0.12.0 with some additional functions.  

## Tutorials

* 1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/Xrenya/pytorch-seq2seq/blob/main/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xrenya/pytorch-seq2seq/blob/main/1_Sequence_to_Sequence_Learning_with_Neural_Networks.ipynb)

    This first tutorial covers the workflow of a PyTorch with torchtext seq2seq project. We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these models in PyTorch, and how to use torchtext to do all of the heavy lifting with regards to text processing. The model itself will be based off an implementation of Sequence to Sequence Learning with Neural Networks, which uses multi-layer LSTMs.

* 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/Xrenya/pytorch-seq2seq/blob/main/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_ipynb.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xrenya/pytorch-seq2seq/blob/main/2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_ipynb.ipynb)
    
    Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch and torchtext gained from the previous tutorial, we'll cover a second second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

* 3 - [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/Xrenya/pytorch-seq2seq/blob/main/3_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xrenya/pytorch-seq2seq/blob/main/3_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb)

    Next, we learn about attention by implementing [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This further allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.

* 4 - [Packed Padded Sequences, Masking, Inference and BLEU](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)

    In this notebook, we will improve the previous model architecture by adding *packed padded sequences* and *masking*. These are two methods commonly used in NLP. Packed padded sequences allow us to only process the non-padded elements of our input sentence with our RNN. Masking is used to force the model to ignore certain elements we do not want it to look at, such as attention over padded elements. Together, these give us a small performance boost. We also cover a very basic way of using the model for inference, allowing us to get translations for any sentence we want to give to the model and how we can view the attention values over the source sequence for those translations. Finally, we show how to calculate the BLEU metric from our translations.

* 5 - [Convolutional Sequence to Sequence Learning](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)

    We finally move away from RNN based models and implement a fully convolutional model. One of the downsides of RNNs is that they are sequential. That is, before a word is processed by the RNN, all previous words must also be processed. Convolutional models can be fully parallelized, which allow them to be trained much quicker. We will be implementing the [Convolutional Sequence to Sequence](https://arxiv.org/abs/1705.03122) model, which uses multiple convolutional layers in both the encoder and decoder, with an attention mechanism between them.  

* 6 - [Attention Is All You Need](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)

    Continuing with the non-RNN based models, we implement the Transformer model from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This model is based soley on attention mechanisms and introduces Multi-Head Attention. The encoder and decoder are made of multiple layers, with each layer consisting of Multi-Head Attention and Positionwise Feedforward sublayers. This model is currently used in many state-of-the-art sequence-to-sequence and transfer learning tasks.

## Reference
1. https://github.com/bentrevett/pytorch-seq2seq
2. https://github.com/spro/practical-pytorch
3. https://github.com/keon/seq2seq
4. https://github.com/pengshuang/CNN-Seq2Seq
5. https://github.com/pytorch/fairseq
6. https://github.com/jadore801120/attention-is-all-you-need-pytorch
7. http://nlp.seas.harvard.edu/2018/04/03/attention.html
8. https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
