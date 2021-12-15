import os
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn

from onmt.utils.distributed import is_master
from onmt.utils.logging import logger
from onmt.utils.module_splitter import explode_model


def build_model_saver(model_opt, opt, model, fields, optim, output_id):
    model_saver = ModelSaver(
        opt.save_model, model, model_opt, fields, optim, opt.keep_checkpoint, output_id
    )
    return model_saver


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(
        self,
        base_path,
        model,
        model_opt,
        fields,
        optim,
        keep_checkpoint=-1,
        output_id="0",
    ):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
        self.output_id = output_id

    def save(self, step, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        if moving_average:
            save_model = deepcopy(self.model)
            for avg, param in zip(moving_average, save_model.parameters()):
                param.data.copy_(avg.data)
        else:
            save_model = self.model

        chkpt_names = self._save(step, save_model, self.output_id)
        self.last_saved_step = step

        if moving_average:
            del save_model

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_names)

    def _save(self, step):
        """Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model, output_id):
        real_model = model.module if isinstance(model, nn.DataParallel) else model
        # real_generator = (real_model.generator.module
        #                  if isinstance(real_model.generator, nn.DataParallel)
        #                  else real_model.generator)
        model_state_dict = real_model.state_dict()

        checkpoint = {
            "model": model_state_dict,
            # 'generator': generator_state_dict,
            "vocab": self.fields,
            "opt": self.model_opt,
            "optim": self.optim.state_dict(),
            "whole_model": self.model,
        }

        checkpoint_paths = []

        if is_master(output_id):
            logger.info("Saving full checkpoint %s_step_%d.pt" % (self.base_path, step))
            checkpoint_path = "%s_step_%d.pt" % (self.base_path, step)
            torch.save(checkpoint, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

        encoders, decoders, attention_bridge, generators, model_frame = explode_model(
            checkpoint
        )

        # TODO: refactor (in a dedicated saver class?)
        # TODO: file names should contain languages instead of device id and enc/dec number, and do not store duplicates
        # encoder modules
        for i, encoder in enumerate(encoders):
            checkpoint_path = "{}_device_{}_step_{}_encoder_{}.pt".format(
                self.base_path, output_id, step, i
            )
            logger.info("Saving encoder checkpoint {}".format(checkpoint_path))
            torch.save(encoder, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
        # decoder modules
        for i, decoder in enumerate(decoders):
            checkpoint_path = "{}_device_{}_step_{}_decoder_{}.pt".format(
                self.base_path, output_id, step, i
            )
            logger.info("Saving decoder checkpoint {}".format(checkpoint_path))
            torch.save(decoder, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
        # generator modules
        for i, generator in enumerate(generators):
            checkpoint_path = "{}_device_{}_step_{}_generator_{}.pt".format(
                self.base_path, output_id, step, i
            )
            logger.info("Saving generator checkpoint {}".format(checkpoint_path))
            torch.save(generator, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

        if is_master(output_id):
            # TODO: not sure how to deal with model_state_dict, fields, model_opt and optim.state_dict() in a multi-gpu
            #  setting. Is it OK to save only from master?
            # attention bridge module
            checkpoint_path = "{}_step_{}_bridge.pt".format(self.base_path, step)
            logger.info("Saving attention bridge checkpoint {}".format(checkpoint_path))
            torch.save(attention_bridge, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

            # model frame
            checkpoint_path = "{}_step_{}_frame.pt".format(self.base_path, step)
            logger.info("Saving model frame checkpoint {}".format(checkpoint_path))
            torch.save(model_frame, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

        return checkpoint_paths

    def _rm_checkpoint(self, names):
        for name in names:
            os.remove(name)
