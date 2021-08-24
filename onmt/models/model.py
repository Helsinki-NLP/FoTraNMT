""" Onmt NMT Model base class definition """
import torch.nn as nn

from onmt.attention_bridge import AttentionBridge
import onmt

class MultiTaskModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoders, decoders, model_opt, multigpu=False):
        super(MultiTaskModel, self).__init__()
        self.multigpu = multigpu
        
        encoder_ids = {lang_code: idx
                   for lang_code, idx
                   in zip(encoders.keys(), range(len(list(encoders.keys()))))}
        encoders = nn.ModuleList(encoders.values())
        self.encoder_ids = encoder_ids
        self.encoders = encoders

        decoder_ids = {lang_code: idx
                       for lang_code, idx
                       in zip(decoders.keys(), range(len(list(decoders.keys()))))}
        dectypes = [model_opt.decoder_type]*len(list(decoders.keys())) if isinstance(model_opt.decoder_type,str) else model_opt.decoder_type
        decoder_types = {lang_code: dectype 
                        for lang_code, dectype
                        in zip(decoders.keys(), dectypes) }
        self.decoder_types = decoder_types  
        
        decoders = nn.ModuleList(decoders.values())
        self.decoder_ids = decoder_ids
        self.decoders = decoders

        self.use_attention_bridge = model_opt.use_attention_bridge
        self.init_decoder = model_opt.init_decoder

        if self.use_attention_bridge:
            model_opt.word_padding_idx = self.encoders[0].embeddings.word_padding_idx
            self.attention_bridge = AttentionBridge(model_opt.rnn_size, model_opt.attention_heads, model_opt)

            # Layers shared across language family
            fam_ids = {famcode:idx for idx,famcode in enumerate(set(model_opt.lang_fam))}
            if model_opt.lang_fam[0]:
                encoder_fams = {lpair.split('-')[0]:fam_ids[model_opt.lang_fam[i]] for i, lpair in enumerate(model_opt.src_tgt)}  
                fam_bridges = nn.ModuleList([AttentionBridge(model_opt.rnn_size, model_opt.attention_heads, model_opt) for _ in fam_ids])
                if fam_ids.get('None'):
                    fam_bridges[fam_ids['None']] = nn.Identity()
            else:
                encoder_fams = {lpair.split('-')[0]:0 for i, lpair in enumerate(model_opt.src_tgt)}  
                fam_bridges = nn.ModuleList([nn.Identity()])

            # Layers shared across language group
            grp_ids = {grpcode:idx for idx,grpcode in enumerate(set(model_opt.lang_group))}
            if model_opt.lang_group[0]:
                encoder_grps = {lpair.split('-')[0]:grp_ids[model_opt.lang_group[i]] for i, lpair in enumerate(model_opt.src_tgt)}  
                grp_bridges = nn.ModuleList([AttentionBridge(model_opt.rnn_size, model_opt.attention_heads, model_opt) for _ in grp_ids])
                if grp_ids.get('None'):
                    grp_bridges[grp_ids['None']] = nn.Identity()
            else:
                encoder_grps = {lpair.split('-')[0]:0 for i, lpair in enumerate(model_opt.src_tgt)}  
                grp_bridges = nn.ModuleList([nn.Identity()])
            
            self.encoder_fams = encoder_fams
            self.fam_bridges = fam_bridges
            self.encoder_grps = encoder_grps
            self.grp_bridges = grp_bridges
        
        # generator ids is linked with decoder_ids
        # self.generators = None

    def forward(self, src, tgt, src_task, tgt_task, lengths, dec_state=None, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):
                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        encoder = self.encoders[self.encoder_ids[src_task]]
        decoder = self.decoders[self.decoder_ids[tgt_task]]
        
        enc_final, memory_bank, lengths = encoder(src, lengths)

        #TEST
        alphas = None
        if self.use_attention_bridge:
                
            alphas, memory_bank = self.grp_bridges[self.encoder_grps[src_task]]((src, memory_bank))
            alphas, memory_bank = self.fam_bridges[self.encoder_fams[src_task]]((src, memory_bank))

            alphas, memory_bank = self.attention_bridge((src, memory_bank))
            
            if self.decoder_types[tgt_task]=='transformer':
                enc_state = decoder.init_state(memory_bank, memory_bank, enc_final)
            else: 
                enc_state = decoder.init_state(src, memory_bank, enc_final) 
        else:
            if self.decoder_types[tgt_task]=='transformer' and isinstance(encoder, 
                (onmt.encoders.audio_encoder.AudioEncoder,onmt.encoders.audio_encoder.AudioEncoderTrf,onmt.encoders.audio_encoder.AudioEncoderTrfv1)):
                #src4init=memory_bank
                enc_state = decoder.init_state(memory_bank, memory_bank, enc_final) 
            else:
                enc_state = decoder.init_state(src, memory_bank, enc_final) 




        # Implement attention bridge/compound attention
        #if self.use_attention_bridge:
        #    alphas, memory_bank = self.attention_bridge(memory_bank, src)

        # for transformer decoders, init state with correct size 
        # (only used for masking, not needed with attBridge)
        #if self.use_attention_bridge and self.decoder_types[tgt_task]=='transformer':
        #    enc_state = \
        #        decoder.init_state(memory_bank, memory_bank, enc_final)
        #else:
        #    enc_state = \
        #        decoder.init_state(src, memory_bank, enc_final)    

        decoder_outputs, attns = \
            decoder(tgt, memory_bank, memory_lengths=lengths)

        #decoder_outputs, dec_state, attns = \
        #    decoder(tgt, memory_bank,
        #            enc_state if dec_state is None
        #            else dec_state,
        #            memory_lengths=lengths)

        #if self.multigpu:
        #    # Not yet supported on multi-gpu
        #    dec_state = None
        #    attns = None
        return decoder_outputs, attns, alphas


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
