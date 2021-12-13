import os
import torch
import torch.nn as nn

from collections import deque

from onmt.utils.distributed import is_master
from onmt.utils.logging import logger

from copy import deepcopy


def build_model_saver(model_opt, opt, model, fields, optim, output_id):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.keep_checkpoint,
                             output_id)
    return model_saver


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 keep_checkpoint=-1, output_id="0"):
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
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        #real_generator = (real_model.generator.module
        #                  if isinstance(real_model.generator, nn.DataParallel)
        #                  else real_model.generator)
        checkpoint_paths = []

        # TODO: file names should contain languages instead of device id and enc/dec number, and do not store duplicates
        for i, encoder in enumerate(self.model.encoders):
            enc_chkpt = {
                'encoder_{}'.format(i): encoder
            }
            enc_chkpt_path = "{}_device{}_encoder{}_step{}.pt".format(self.base_path, output_id, i,  step)
            logger.info("Saving encoder {}".format(enc_chkpt_path))
            torch.save(enc_chkpt, enc_chkpt_path)
            checkpoint_paths.append(enc_chkpt_path)

        for i, decoder in enumerate(self.model.decoders):
            dec_chkpt = {
                'decoder_{}'.format(i): decoder
            }
            dec_chkpt_path = "{}_device{}_decoder{}_step{}.pt".format(self.base_path, output_id, i, step)
            logger.info("Saving decoder {}".format(dec_chkpt_path))

            torch.save(dec_chkpt, dec_chkpt_path)
            checkpoint_paths.append(dec_chkpt_path)

        if is_master(output_id):
            # TODO: not sure how to deal with model_state_dict, fields, model_opt and optim.state_dict() in a multi-gpu
            #  setting. Is it OK to save only from master?
            model_state_dict = real_model.state_dict()
            # model_state_dict = {k: v for k, v in model_state_dict.items()
            #                    if 'generator' not in k}
            # generator_state_dict = real_generator.state_dict()
            att_chkpt = {
                'model': model_state_dict,
                # 'generator': generator_state_dict,
                'vocab': self.fields,
                'opt': self.model_opt,
                'optim': self.optim.state_dict(),
                'attention_bridge': self.model.attention_bridge
            }

            att_chkpt_path = "{}_bridge_step{}.pt".format(self.base_path, step)

            logger.info("MASTER: Saving attention_bridge {}".format(att_chkpt_path))
            torch.save(att_chkpt, att_chkpt_path)
            checkpoint_paths.append(att_chkpt_path)

        return checkpoint_paths

    def _rm_checkpoint(self, names):
        for name in names:
            os.remove(name)
