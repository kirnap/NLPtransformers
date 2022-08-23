"""
Week 1 implementations:

http://nlp.seas.harvard.edu/annotated-transformer/
"""
import copy

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad


class EncoderDecoder(nn.Module):
    """
    Encoder maps an input sequence of symbol representations (x_1, .., x_n) to a sequence of
    continuous representations (z_1, ..., z_n). Given Z the decoder then generates an output
    sequence (y_1, ..., y_m) of symbols one element at a time.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 src_embed,
                 tgt_embed,
                 generator
                 ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        def forward(self, src, tgt, src_mask,tgt_mask):
            """forward step of the model"""
            return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

        def encode(self, src, src_mask):
            return self.encoder(self.src_embed(src), src_mask)

        def decode(self, memory, src_mask, tgt, tgt_mask):
            """memory is the encoded sequence taken from encoder"""
            return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """To generate output"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    """
    Layernorm module.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Here is an example pytorch code:
        let x be a torch.tensor:

        x
        tensor([[0.9434, 0.0042, 0.6894],
                [0.1830, 0.7708, 0.5469]])

        x.mean(-1, keepdim=True) returns -->
        tensor([[0.5457],
                [0.5002]])
        Same for std too
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std+self.eps) + self.b_2


def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feedforward"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


######### Decoder #########

# Decoder inserts multi-head attention over the output of encoder stack.

class Decoder(nn.Module):
    """Generic N layer decoder with masking"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Memory is the output of encoder"""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory  # TODO: why do we have this?
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    pass  # TODO: continue from here