# FoTraNMT - Scaling Up Language Coverage in multilingual Neural Machine Translation

FotraNMT is a multilingual NMT toolkit developed as part of the [FoTran project](http://www.helsinki.fi/fotran) at University of Helsinki.
We developed FoTraNMT specifically to train and extract massively multilingual meaning representations in a cost-effective way. It includes features for multilingual & multimodal machine translation enabled by distributed training of independent encoders and decoders connected via an inner-attention layer. 

FoTraNMT is optimized for training large models (on a sufficiently large high-performance cluster). For this, we pay special attention at 
1. distributing the modules across several processing units, and 
2. efficiently training the network over many translation direction.  

## This branch: `who-would-win`

This branch specifically covers experiments detailed in the NoDaLiDa 2023 paper ["Dozens of Translation Directions or Millions of Shared Parameters? Comparing Two Types of Multilinguality in Modular Machine Translation"](https://openreview.net/forum?id=1vkyEY-HeLY). 
This paper studies the effect of different amounts of parameter sharing and number of languages on encoder representations.
This branch contains scripts to replicate training on a slurm cluster (e.g., CSC puhti or mahti), as well as scripts to replicate experiments described in the paper.

## Requirements & Installation 
You need to clone this repo
```bash
git clone --branch who-would-win https://github.com/Helsinki-NLP/FoTraNMT.git
cd FoTraNMT
```
We strongly recommend to make the setup in a virtual environment. 
This is done by:
```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
```
_You can confirm you’re in the virtual environment by checking the location of your Python interpreter, it should point to the env directory:_
```bash
which python
pip install --upgrade pip
```
Now that you’re in your virtual environment you can install packages without worrying too much about version control.

Packages necessary should be listed in the `setup.py` file. Assuming your current working directory corresponds to the top-level directory of this git, you should be able to install all dependencies with:
```bash
python3 -m pip install . -e
```


## Citing this work

Please cite the corresponding paper:
```latex
@inproceedings{boggia2023dozens,
   title={Dozens of Translation Directions or Millions of Shared Parameters? Comparing Two Types of Multilinguality in Modular Machine Translation},
   author={Michele Boggia and 
      Stig-Arne Gr{\"o}nroos and 
      Niki Andreas Loppi and 
      Timothee Mickus and 
      Alessandro Raganato and 
      J{\"o}rg Tiedemann and 
      Ra{\'u}l V{\'a}zquez
   },
   booktitle={The 24rd Nordic Conference on Computational Linguistics},
   year={2023},
   url={https://openreview.net/forum?id=1vkyEY-HeLY}
}
```

Also consider citing the work it builds up on:
[Introduction to the architecture](https://www.aclweb.org/anthology/2020.cl-2.5)
```latex
@article{vazquez-etal-2020-systematic,
    title = "A Systematic Study of Inner-Attention-Based Sentence Representations in Multilingual Neural Machine Translation",
    author = {V{\'a}zquez, Ra{\'u}l  and
      Raganato, Alessandro  and
      Creutz, Mathias  and
      Tiedemann, J{\"o}rg},
    journal = "Computational Linguistics",
    volume = "46",
    number = "2",
    month = jun,
    year = "2020",
    url = "https://aclanthology.org/2020.cl-2.5",
    doi = "10.1162/coli_a_00377",
    pages = "387--424",
}

```    

FoTraNMT is built on top of OpenNMT, so please also acknowledge them if you use the system. See the [OpenNMT-py offical repo](https://github.com/OpenNMT/OpenNMT-py) and [website](https://opennmt.net/) for documentation of the system's basic functionalities.
```
@inproceedings{klein-etal-2017-opennmt,
    title = "{O}pen{NMT}: Open-Source Toolkit for Neural Machine Translation",
    author = "Klein, Guillaume  and
      Kim, Yoon  and
      Deng, Yuntian  and
      Senellart, Jean  and
      Rush, Alexander",
    booktitle = "Proceedings of {ACL} 2017, System Demonstrations",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-4012",
    pages = "67--72",
}
```

