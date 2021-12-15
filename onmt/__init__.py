""" Main entry point of the ONMT library """
from __future__ import division, print_function

import sys

import onmt.decoders
import onmt.encoders
import onmt.inputters
import onmt.models
import onmt.modules
import onmt.opts
import onmt.utils
import onmt.utils.optimizers
from onmt.trainer import Trainer

onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [onmt.inputters, onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules, "Trainer", onmt.opts]

__version__ = "0.9.0"
