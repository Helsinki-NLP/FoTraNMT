import argparse

import torch

import onmt
import translate
from onmt.utils.module_splitter import build_bilingual_model

if __name__ == "__main__":
    parser = translate.ArgumentParser(
        description="Build bilingual model putting together module checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # onmt.opts.translate_opts(parser)
    onmt.opts.build_bilingual_model(parser)

    opt = parser.parse_args()

    encoder = torch.load(opt.encoder, map_location=lambda storage, loc: storage)
    decoder = torch.load(opt.decoder, map_location=lambda storage, loc: storage)
    bridge = torch.load(opt.bridge, map_location=lambda storage, loc: storage)
    generator = torch.load(opt.generator, map_location=lambda storage, loc: storage)
    frame = torch.load(opt.model_frame, map_location=lambda storage, loc: storage)

    recombined_bilingual = build_bilingual_model(
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        enc_module=encoder,
        dec_module=decoder,
        ab_module=bridge,
        gen_module=generator,
        model_frame=frame,
    )

    torch.save(recombined_bilingual, opt.output)
