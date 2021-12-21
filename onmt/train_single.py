#!/usr/bin/env python
"""Training on a single process."""
import os
from collections import OrderedDict

import numpy as np
import torch

from onmt.inputters.inputter import build_dataset_iter, load_old_vocab, old_style_vocab
from onmt.model_builder import (
    build_model,
    build_embeddings_then_encoder,
    build_decoder_and_generator,
)
from onmt.models import build_model_saver
from onmt.trainer import build_trainer
from onmt.utils.distributed import is_master, CommunicationGroup
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed
from onmt.utils.optimizers import Optimizer
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if "encoder" in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def build_dataset_iter_fct(dataset_name, fields_, opt_, data_path, is_train=True):
    def train_iter_wrapper():
        return build_dataset_iter(dataset_name, fields_, opt_, data_path, is_train)

    return train_iter_wrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def update_to_local_attr(attribute, index):
    """return local attribure for encoders and decoders"""
    if type(attribute) is list:
        if len(attribute) > 1:
            attr = attribute[index]
        else:
            attr = attribute[0]
    else:
        attr = attribute
    return attr


def main(opt, unique_device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    device_id = unique_device_id % 4

    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(
        opt.accum_steps
    ), "Number of accum_count values must match number of accum_steps"
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info("Loading checkpoint from %s" % opt.train_from)
        checkpoint = torch.load(
            opt.train_from, map_location=lambda storage, loc: storage
        )

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info("Loading vocab from checkpoint at %s." % opt.train_from)
        vocab = checkpoint["vocab"]
    else:
        checkpoint = None
        model_opt = opt

    train_iters = OrderedDict()
    valid_iters = OrderedDict()

    encoders = OrderedDict()
    decoders = OrderedDict()

    generators = OrderedDict()
    src_vocabs = OrderedDict()
    tgt_vocabs = OrderedDict()
    fields_dict = OrderedDict()

    # variables needed for sharing the same embedding matrix across encoders and decoders
    first_time = True
    weight_to_share = None

    # Create a lookup list for the GPU-allocations
    num_pairs = len(opt.src_tgt)
    pairs_per_gpu = num_pairs // opt.world_size
    gpu_alloc_idx = []
    for i in range(opt.world_size):
        gpu_alloc_idx += [i] * pairs_per_gpu
    if is_master(unique_device_id):
        logger.info("Pairs: {}".format(opt.src_tgt))
        logger.info("gpu_alloc_indices: {}".format(gpu_alloc_idx))
    # Empty lists to track encoder/decoder names on all gpus
    encoder_list = []
    decoder_list = []

    # we share the word embedding space when source lang and target lang are the same
    map_lang2_emb = {}
    # for (src_tgt_lang), data_path in zip(opt.src_tgt, opt.data):
    for index in range(len(opt.src_tgt)):
        src_tgt_lang = opt.src_tgt[index]
        data_path = opt.data[index]
        local_enc_dec_opts = AttrDict(
            {key: model_opt.__dict__[key] for key in model_opt.__dict__.keys()}
        )
        local_enc_dec_opts.model_type = update_to_local_attr(
            model_opt.model_type, index
        )
        local_enc_dec_opts.audio_enc_pooling = update_to_local_attr(
            model_opt.audio_enc_pooling, index
        )
        local_enc_dec_opts.n_mels = update_to_local_attr(model_opt.n_mels, index)
        local_enc_dec_opts.n_stacked_mels = update_to_local_attr(
            model_opt.n_stacked_mels, index
        )
        local_enc_dec_opts.enc_layers = update_to_local_attr(
            model_opt.enc_layers, index
        )
        local_enc_dec_opts.dec_layers = update_to_local_attr(
            model_opt.dec_layers, index
        )
        local_enc_dec_opts.rnn_type = update_to_local_attr(model_opt.rnn_type, index)
        local_enc_dec_opts.encoder_type = update_to_local_attr(
            model_opt.encoder_type, index
        )
        local_enc_dec_opts.batch_size = update_to_local_attr(opt.batch_size, index)
        local_enc_dec_opts.batch_type = update_to_local_attr(opt.batch_type, index)
        local_enc_dec_opts.normalization = update_to_local_attr(
            model_opt.normalization, index
        )

        src_lang, tgt_lang = src_tgt_lang.split("-")

        vocab = torch.load(data_path + ".vocab.pt")

        # check for code where vocab is saved instead of fields
        # (in the future this will be done in a smarter way)
        if old_style_vocab(vocab):
            fields = load_old_vocab(
                vocab, opt.model_type[0], dynamic_dict=opt.copy_attn
            )
        else:
            fields = vocab

        if is_master(unique_device_id):
            # Report src and tgt vocab sizes, including for features
            for (side, lang_code) in [("src", src_lang), ("tgt", tgt_lang)]:
                f = fields[side]
                try:
                    f_iter = iter(f)
                except TypeError:
                    f_iter = [(side, f)]
                for sn, sf in f_iter:
                    if sf.use_vocab:
                        logger.info(
                            " * {} ({}) vocab size = {}".format(
                                sn, lang_code, len(sf.vocab)
                            )
                        )

        # Build model.
        encoder, src_embeddings = build_embeddings_then_encoder(
            local_enc_dec_opts, fields
        )

        # Consider only encoder corresponding to the rank
        if gpu_alloc_idx[index] == unique_device_id:
            encoders[src_lang] = encoder

        decoder, generator, tgt_embeddings = build_decoder_and_generator(
            local_enc_dec_opts, fields
        )

        # Consider only decoder corresponding to the rank
        if gpu_alloc_idx[index] == unique_device_id:
            decoders[tgt_lang] = decoder

        # Share the embedding matrix across all the encoders and decoders - preprocess with share_vocab required.
        if model_opt.share_embeddings and first_time:
            tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight
            weight_to_share = src_embeddings.word_lut.weight
        if model_opt.share_embeddings and (not first_time):
            tgt_embeddings.word_lut.weight = weight_to_share
            src_embeddings.word_lut.weight = weight_to_share
        first_time = False

        if src_lang in map_lang2_emb and model_opt.model_type == "text":
            encoder.embeddings.word_lut.weight = map_lang2_emb.get(src_lang)
        elif model_opt.model_type == "text":
            map_lang2_emb[src_lang] = src_embeddings.word_lut.weight
        if tgt_lang in map_lang2_emb:
            decoder.embeddings.word_lut.weight = map_lang2_emb.get(tgt_lang)
        else:
            map_lang2_emb[tgt_lang] = tgt_embeddings.word_lut.weight

        if model_opt.model_type == "text":
            src_vocabs[src_lang] = fields["src"].base_field.vocab
        tgt_vocabs[tgt_lang] = fields["tgt"].base_field.vocab

        if gpu_alloc_idx[index] == unique_device_id:
            generators[tgt_lang] = generator

        if gpu_alloc_idx[index] == unique_device_id:
            # add this dataset iterator to the training iterators
            train_iters[(src_lang, tgt_lang)] = build_dataset_iter_fct(
                "train", fields, data_path, local_enc_dec_opts
            )
            # add this dataset iterator to the validation iterators
            valid_iters[(src_lang, tgt_lang)] = build_dataset_iter_fct(
                "valid", fields, data_path, local_enc_dec_opts, is_train=False
            )

        fields_dict[src_tgt_lang] = fields

        # Track encoders and decoder names on all ranks to build communicators
        encoder_list.append(src_lang)
        decoder_list.append(tgt_lang)

    encoder_splits = np.split(np.array(encoder_list), opt.world_size)
    decoder_splits = np.split(np.array(decoder_list), opt.world_size)

    all_enc_comms = OrderedDict()
    for l in sorted(set(encoder_list)):
        indices = [i for i, x in enumerate(encoder_splits) if l in x]
        if len(indices) > 1:
            if is_master(unique_device_id):
                logger.info("Enc comm group {} {}".format(l, indices))
            all_enc_comms[l] = CommunicationGroup(
                torch_dist_group=torch.distributed.new_group(indices),
                group_size=len(indices),
            )

    all_dec_comms = OrderedDict()
    for l in sorted(set(decoder_list)):
        indices = [i for i, x in enumerate(decoder_splits) if l in x]
        if (
            len(indices) > 1
        ):  # maybe needs to add not in logic to remove duplicate comms
            if is_master(unique_device_id):
                logger.info("Dec comm group {} {}".format(l, indices))
            all_dec_comms[l] = CommunicationGroup(
                torch_dist_group=torch.distributed.new_group(indices),
                group_size=len(indices),
            )

    # Build model.
    model = build_model(
        model_opt,
        opt,
        fields,
        encoders,
        decoders,
        generators,
        src_vocabs,
        tgt_vocabs,
        checkpoint,
    )

    n_params, enc, dec = _tally_parameters(model)
    logger.info(f"total encoder parameters: {enc}")
    logger.info(f"total encodes: {len(model.encoders)}")
    for k, v in model.encoder_ids.items():
        n_current, _, _ = _tally_parameters(model.encoders[v])
        logger.info(
            f"[GPU ranks {unique_device_id},{device_id}] Enc [{v}]= name: {k}, params: {n_current}"
        )
    for k, v in model.decoder_ids.items():
        n_current, _, _ = _tally_parameters(model.decoders[v])
        logger.info(
            f"[GPU ranks {unique_device_id},{device_id}] Dec [{v}]= name: {k}, params: {n_current}"
        )
    # logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optimizer = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(
        model_opt, opt, model, fields_dict, optimizer, unique_device_id
    )

    trainer = build_trainer(
        opt,
        unique_device_id,
        gpu_alloc_idx,
        all_enc_comms,
        all_dec_comms,
        model,
        fields,
        optimizer,
        generators,
        tgt_vocabs,
        model_saver=model_saver,
    )

    # TODO: not implemented yet
    # train_iterables = []
    # if len(opt.data_ids) > 1:
    #    for train_id in opt.data_ids:
    #        shard_base = "train_" + train_id
    #        iterable = build_dataset_iter(shard_base, fields, opt, multi=True)
    #        train_iterables.append(iterable)
    #    train_iter = MultipleDatasetIterator(train_iterables, device_id, opt)
    # else:
    #    train_iter = build_dataset_iter("train", fields, opt)
    # valid_iter = build_dataset_iter(
    #    "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info("Starting training on GPU: %s" % opt.gpu_ranks)
    else:
        logger.info("Starting training on CPU, could be very slow")
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_iters,
        train_steps,
        opt.save_checkpoint_steps,
        valid_iters,
        opt.valid_steps,
    )

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()
