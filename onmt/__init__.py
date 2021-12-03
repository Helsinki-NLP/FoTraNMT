""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.inputters
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules
from onmt.trainer import Trainer
import sys
import onmt.utils.optimizers
import onmt.opts
onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [onmt.inputters, onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules, "Trainer", onmt.opts]

__version__ = "0.9.0"
