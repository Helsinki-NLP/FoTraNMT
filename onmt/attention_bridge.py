"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt

class AttentionBridgeLayer(nn.Module):
    """
    Multi-headed attention. Bridge between encoders->decoders
    """
    def __init__(self, hidden_size, attention_heads, model_opt):
        """Attention Heads Layer:"""
        super(AttentionBridgeLayer, self).__init__()
        d = hidden_size
        u = model_opt.hidden_ab_size
        r = attention_heads
        #TEST
        self.model_type = model_opt.model_type
        if self.model_type != "text":
            d = model_opt.dec_rnn_size
        self.ws1 = nn.Linear(d, u, bias=True)
        self.ws2 = nn.Linear(u, r, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = r
        #self.layer_norm = nn.LayerNorm(d, eps=1e-6)
        self.M = None


    def forward(self, enc_in_out: tuple):
        enc_input, enc_output = enc_in_out
        self.M, alphas = self.mixAtt(enc_output, enc_input)

        #output, alphas = self.mixAtt(enc_output, enc_input)
        #take transpose to match dimensions s.t. r=new_seq_len:
        #self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        #self.M = self.layer_norm(self.M)
        #h_avrg = (self.M).mean(dim=0, keepdim=True)
        return alphas, self.M


    def mixAtt(self, outp, inp):
        """Notation based on Lin et al. (2017) A structured self-attentive sentence embedding"""
        #outp = torch.transpose(outp, 0, 1).contiguous() # <- passed to the AttentionBridge
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]

        hbar = self.relu(self.ws1(compressed_embeddings))  # [bsz*len, attention-unit]

        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        #Penalize alphas if "text"
        if self.model_type == "text":
            transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
            transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
            concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
            concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

            penalized_alphas = alphas + (-10000 * (concatenated_inp == 1).float()) # [bsz, hop, len] + [bsz, hop, len]
            alphas = penalized_alphas

        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

class AttentionBridge(nn.Module):
    """
    N-layered attention-bridge between encoders->decoders
    """
    def __init__(self, hidden_size, attention_heads, model_opt):
        """Attention Heads Layer:"""
        super(AttentionBridge, self).__init__()
        self.ab_nlayers = model_opt.n_layers_attbrg 
        self.ab_layertype = model_opt.layer_type_attbrg 
        self.word_padding_idx = model_opt.word_padding_idx
        if self.ab_layertype == 'fixed-size':
            self.attbrg = nn.ModuleList(
                [AttentionBridgeLayer(
                    hidden_size, attention_heads, model_opt)
                for i in range(self.ab_nlayers)])
        elif self.ab_layertype == 'transformer':
            dropout = model_opt.dropout[0] if type(model_opt.dropout) is list else model_opt.dropout
            from  onmt.encoders.transformer import TransformerEncoderLayer
            self.attbrg = nn.ModuleList(
                [TransformerEncoderLayer(
                    d_model=model_opt.enc_rnn_size, 
                    heads=model_opt.heads, 
                    d_ff=model_opt.transformer_ff, 
                    dropout=dropout,
                    max_relative_positions=model_opt.max_relative_positions)
                for i in range(self.ab_nlayers-1)])
            self.attbrg.append(
                AttentionBridgeLayer(
                    hidden_size, attention_heads, model_opt)
                    )
    
    def forward(self, enc_in_out):
        """Forward pass for the bridge layers"""
        
        src, out = enc_in_out
        out = torch.transpose(out, 0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        mask = words.data.eq(self.word_padding_idx).unsqueeze(1)  # [B, 1, T]
        for layer in self.attbrg:
            if isinstance(layer, AttentionBridgeLayer):
                alphas, out = layer((src,out)) 
            else:
                out = layer(out, mask)
        
        return alphas, out.transpose(0, 1).contiguous()
