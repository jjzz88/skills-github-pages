<center><h1>The Annotated Transformer</h1> </center>


<center>
<p><a href="https://arxiv.org/abs/1706.03762">Attention is All You Need
</a></p>
</center>

* *v2022: Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak,
   and Stella Biderman.*
* *[Original](https://nlp.seas.harvard.edu/2018/04/03/attention.html):
   [Sasha Rush](http://rush-nlp.com/).*


The Transformer has been on a lot of
people's minds over the last <s>year</s> five years.
This post presents an annotated version of the paper in the
form of a line-by-line implementation. It reorders and deletes
some sections from the original paper and adds comments
throughout. This document itself is a working notebook, and should
be a completely usable implementation.
Code is available
[here](https://github.com/harvardnlp/annotated-transformer/).


<h3> Table of Contents </h3>
<ul>
<li><a href="#prelims">Prelims</a></li>
<li><a href="#background">Background</a></li>
<li><a href="#part-1-model-architecture">Part 1: Model Architecture</a></li>
<li><a href="#model-architecture">Model Architecture</a><ul>
<li><a href="#encoder-and-decoder-stacks">Encoder and Decoder Stacks</a></li>
<li><a href="#position-wise-feed-forward-networks">Position-wise Feed-Forward
Networks</a></li>
<li><a href="#embeddings-and-softmax">Embeddings and Softmax</a></li>
<li><a href="#positional-encoding">Positional Encoding</a></li>
<li><a href="#full-model">Full Model</a></li>
<li><a href="#inference">Inference:</a></li>
</ul></li>
<li><a href="#part-2-model-training">Part 2: Model Training</a></li>
<li><a href="#training">Training</a><ul>
<li><a href="#batches-and-masking">Batches and Masking</a></li>
<li><a href="#training-loop">Training Loop</a></li>
<li><a href="#training-data-and-batching">Training Data and Batching</a></li>
<li><a href="#hardware-and-schedule">Hardware and Schedule</a></li>
<li><a href="#optimizer">Optimizer</a></li>
<li><a href="#regularization">Regularization</a></li>
</ul></li>
<li><a href="#a-first-example">A First Example</a><ul>
<li><a href="#synthetic-data">Synthetic Data</a></li>
<li><a href="#loss-computation">Loss Computation</a></li>
<li><a href="#greedy-decoding">Greedy Decoding</a></li>
</ul></li>
<li><a href="#part-3-a-real-world-example">Part 3: A Real World Example</a>
<ul>
<li><a href="#data-loading">Data Loading</a></li>
<li><a href="#iterators">Iterators</a></li>
<li><a href="#training-the-system">Training the System</a></li>
</ul></li>
<li><a href="#additional-components-bpe-search-averaging">Additional
Components: BPE, Search, Averaging</a></li>
<li><a href="#results">Results</a><ul>
<li><a href="#attention-visualization">Attention Visualization</a></li>
<li><a href="#encoder-self-attention">Encoder Self Attention</a></li>
<li><a href="#decoder-self-attention">Decoder Self Attention</a></li>
<li><a href="#decoder-src-attention">Decoder Src Attention</a></li>
</ul></li>
<li><a href="#conclusion">Conclusion</a></li>
</ul>

# Prelims

<a href="#background">Skip</a>


```python
# !pip install -r requirements.txt
```


```python
# # Uncomment for colab
# #
# !pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
# !python -m spacy download de_core_news_sm
# !python -m spacy download en_core_web_sm
```


```python
!pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
```

    [31mERROR: Ignored the following yanked versions: 0.15.0[0m[31m
    [0m[31mERROR: Could not find a version that satisfies the requirement torchtext==0.12 (from versions: 0.1.1, 0.2.0, 0.2.1, 0.2.3, 0.3.1, 0.4.0, 0.5.0, 0.6.0, 0.15.1, 0.15.2, 0.16.0, 0.16.1, 0.16.2, 0.17.0, 0.17.1, 0.17.2, 0.18.0)[0m[31m
    [0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.3.1[0m[39;49m -> [0m[32;49m25.1.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    [31mERROR: No matching distribution found for torchtext==0.12[0m[31m
    [0m
