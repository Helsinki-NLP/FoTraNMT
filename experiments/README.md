# Opus experiments

The plan is to train the model by scaling up the number of translation directions (TDs) and gpus, 
so that the ratio #TD/#GPU is constant.
The aim is inspecting how this affects translation performance (both for supervised and zero-shot directions)
and quality of the internal representation obtained from the shared layer.

In each run of the experiment we train the model on a subset of (non-english) languages from Opus100. 
For each language `X` we train three different directions: `X-en`, `en-X`, `X-X`.
In order to optimize gpu memory and communications we do not include `en-en`, as it would spoil the balance between
devices in terms of #LPs/GPU.

For instance, we can consider the following setup with 9 LPs/GPU (but we can also use 6 LPs/GPU):

|        | Languages | Translation Directions | #GPUs | #LPs/GPU |
|--------|-----------|------------------------|-------|----------|
| opus03 | 3         | 9                      | 1     | 9        |
| opus06 | 6         | 18                     | 2     | 9        |
| opus09 | 9         | 27                     | 3     | 9        |
| opus12 | 12        | 36                     | 4     | 9        |
| opus24 | 24        | 72                     | 8     | 9        |
| opus36 | 36        | 108                    | 12    | 9        |
| opus48 | 48        | 144                    | 16    | 9        |
| opus96 | 96        | 288                    | 32    | 9        |

We also try to keep the number of training examples as balanced as possible.
However, we need to keep in mind that only 44 of the 99 non-English languages in Opus100
have 1M training examples. Thus, when considering more languages, some of them will have a lower amount
of training examples.

Languages are selected following three criteria:
 1. maximise the training signal
 2. possibility to test zero-shot translation
 3. possibility to test over XNLI
 4. maximise language diversity

For the first constraint we sort the languages based on the number of training examples, and start
from those with a larger training set. Statistics on the number of training examples in the Opus100
dataset can be found in `opus100_stats.csv`.  

To satisfy the second condition, we prioritise the following languages: ar, de, fr, nl, ru, zh.
To test over the XNLI data, we give priority to: fr, es, de, el, bg, ru, tr, ar, vi, th, zh, nl
(we exclude 3 low-resource languages that are present in XNLI for which we do not have much training data on Opus100).

We consider the following languages:
 * 3 languages: ar, fr, zh
 * 6 languages: + de, nl, ru
 * 9 languages: + tr, vi, th 
 * 12 languages: + es, el, bg
 * 24 languages: + he, fi, ja, sv, fa, mk, eu, id, bn, ko, it, lv
 * 36 languages: + mt, et, ro, bs, sr, is, uk, hu, lt, cs, sk, sq
 * 44 languages: + ms, da, no, pt, pl, hr, ca, sl
 * 48 languages: + si, ml, ur, mg


## Data preparation

Before training the models, we need to prepare the data.
Our goal is to have a scalable setup, where language-specific encoders and decoders can be added on top of the 
existing ones when necessary.
Also, we want to treat all languages equally.
For these reasons we make use of language-specific vocabularies.
In this way we are not tied to a set of scripts, but we can add any language -- even those with a script that is not in
the original vocabulary.

The on-the-fly data preprocessing of the current system implementation does not support different vocabularies
for different corpora, so we preprocess the data at once before starting our experiments.
We tokenize the data using the sentencepiece models shipped with the 
[OPUS-MT-train](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/tatoeba/SentencePieceModels.md) repository.
Given that at the time of this writing the models are new and should be thoroughly tested, we also prepare our own
sentencepiece models, by training them on the opus100 training data.

The job scripts used for preparing the data are in the `jobs` folder:
 * `job-prepare_opus_tc.sh`: calls the preprocessing script `scripts/prepare_opus_data_tc.mahti.sh`, which 
   downloads the corpus, the sentencepiece models, and prepare the corpus for training and testing
 * `job-prepare_opus.sh`: calls the preprocessing script `scripts/prepare_opus_data.mahti.sh`. Same as above, instead of
   downloading sentencepiece models trains them over the opus data. More details about the steps taken are given in the 
   next section.

In both cases we use the sentencepiece vocabularies (i.e. we do not use the OpenNMT `build_vocabs.py` script).


### Trained Sentencepiece Models and Vocabularies

Before training each sentencepiece model, we deduplicate the training data.
Vocabulary sizes are all set to 32k where possible, and decreased to submultiples of 32k if necessary (i.e. when the
training data has less than 32k tokens).
For Chinese (zh) and Japanese we use a vocabulary size of 64k.

For English, we merge training data from all language pairs included in the corpus, deduplicate it, and sample 10M 
sentences to train the sentencepiece model.

The output vocabularies and models are stored, in Mahti, in the folder
`/scratch/project_2005099/data/opus/prepare_opus_data_out`, named as `opus.<lang>.vocab` and `opus.<lang>.model`.
The vocabularies in the OpenNMT format are stored in the same folder and named `opus.<lang>.vocab.onmt`.


### Data Parsing

We use the trained sentencepiece models to parse all the data available in the Opus100 corpus. Note that for 5 of the
99 languages in the corpus, dev and test data is not available.

The tokenized training files are stored in Mahti in the output folder
`/scratch/project_2005099/data/opus/prepare_opus_data_out`. We use the same folder structure as in the original corpus, 
appending a `.sp` to each file name.


## Scripts

 * `scripts/prepare_opus_data_tc.mahti.sh` downloads data and sentencepiece models and tokenizes the corpus
 * `scripts/prepare_opus_data.mahti.sh` downloads data, train sentencepiece models and tokenizes the corpus
 * `scripts/test_opus.sh` test translation performance of trained models


## Config Files

 * `config/config-opus01-50-adaf-nomono.mahti.yml` config file to test simple scenario with 2 language pairs -- useful for debugging
 * `config/config-opus01-50-adaf-allmono.mahti.yml` config file to test simple scenario with 4 language pairs -- useful for debugging
 * `config/config-opus03-50-adaf.mahti.yml` config file for opus03 training setup using adafactor and and attention bridge with 50 attention heads
 * `config/config-opus06-50-adaf.mahti.yml` config file for opus06 training setup using adafactor and and attention bridge with 50 attention heads


## Job Scripts

 * `jobs/job-prepare_opus_tc.sh` preprocess data using pretrained sentencepiece models from [OPUS-MT-train](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/tatoeba/SentencePieceModels.md)
 * `jobs/job-prepare_opus.sh` preprocess data and train sentencepiece models on the corpus
 * `jobs/job-train_opus03-50-adaf.mahti.sh` train opus03-50-adaf setup (see `config/config-opus03-50-adaf.mahti.yml`)
 * `jobs/job-train_opus06-50-adaf.mahti.sh` train opus06-50-adaf setup (see `config/config-opus06-50-adaf.mahti.yml`)
 * `jobs/job-test_opus03-50-adaf.mahti.sh` test translation performance of opus03-50-adaf setup (see `config/config-opus03-50-adaf.mahti.yml`)


# TODOs

 * data preprocessing: what is the ISO 639-2 of the ISO 639-1 language code `sh`?
   At the moment, the language is skipped in the preprocessing.
 * filter out very long sentences before data preprocessing (see TODO in `scripts/prepare_opus_data_tc.mahti.sh`)
 