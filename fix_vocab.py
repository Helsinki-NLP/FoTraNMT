# -*- coding: utf-8 -*-
import os
import torch
from collections import Counter
import torchtext
import argparse

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def preprocess_opts(parser):
    group = parser.add_argument_group('Options')
    group.add_argument('-l', '-languages', nargs='+', required=True, help="""List of language ids""")
    group.add_argument('-b', '-base_path', required=True, help="""Path to directory containing vocabs""")
    group.add_argument('-v', '-vocab_name', required=True, help="""Common name of the vocabs (<common_name>.<src>-<tgt>.vocab.pt)""")
    group.add_argument('-n', '-new_vocab_path', required=True, help="""Path to existing directory for fixed vocabs""")

def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description="fic_vocab.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    preprocess_opts(parser)

    opt = parser.parse_args()

    return opt

def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab

def AddCounter(lang_tuple, lang, dictLangFreqs):
    Vocab = lang_tuple[1]
    freqs = Vocab.freqs
    c = dictLangFreqs[lang]
    c += freqs
    dictLangFreqs[lang] = c

def fixVocab(langs, basepath, vocabname, basepathNewVocab):
    dictLangFreqs = {}
    for lang in langs:
        dictLangFreqs[lang] = Counter()

    for langSRC in langs:
        for langTGT in langs:
            path=basepath+"/"+vocabname+"."+langSRC+"-"+langTGT+".vocab.pt"
            if os.path.isfile(path):
                vocab_object = torch.load(path)
                # we have stoi and freq
                src_tuple = vocab_object[0]
                AddCounter(src_tuple, langSRC, dictLangFreqs)
                tgt_tuple = vocab_object[1]
                AddCounter(tgt_tuple, langTGT, dictLangFreqs)

    for langSRC in langs:
        for langTGT in langs:
            fields = {}

            fields["src"] = torchtext.legacy.data.Field(
                include_lengths=True)
            fields["tgt"] = torchtext.legacy.data.Field(
                pad_token=PAD_WORD)

            c_tgt = dictLangFreqs[langTGT]
            vocab_tgt = torchtext.vocab.Vocab(c_tgt,
                                  specials=[UNK_WORD, PAD_WORD,
                                            BOS_WORD, EOS_WORD],
                                  max_size=50000)
            fields["tgt"].vocab = vocab_tgt

            c_src = dictLangFreqs[langSRC]
            vocab_src = torchtext.vocab.Vocab(c_src,
                                  specials=[UNK_WORD, PAD_WORD,
                                            BOS_WORD, EOS_WORD],
                                  max_size=50000)
            fields["src"].vocab = vocab_src

            vocab_file = basepathNewVocab+"/"+vocabname+"."+str(langSRC)+"-"+str(langTGT)+".vocab.pt"
            torch.save(save_fields_to_vocab(fields), vocab_file)

if __name__ == "__main__":
    opt = parse_args()
    fixVocab(opt.l, opt.b, opt.v, opt.n)
