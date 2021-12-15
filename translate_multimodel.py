import argparse
import onmt

from ipdb import launch_ipdb_on_exception

import translate

if __name__ == "__main__":
    parser = translate.ArgumentParser(
        description='translate_multimodel.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    # only this line is extra to normal translate.py
    onmt.opts.translate_multimodel(parser)

    opt = parser.parse_args()
    logger = translate.init_logger(opt.log_file)
    with launch_ipdb_on_exception():
        translate.main(opt)
