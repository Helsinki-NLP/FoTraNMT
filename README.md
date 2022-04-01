# FoTraNMT - Scaling Up Language Coverage in multilingual Neural Machine Translation

FotraNMT is a multilingual NMT toolkit developed as part of the [FoTran project](http://www.helsinki.fi/fotran) at University of Helsinki.
We developed FoTraNMT specifically to train and extract massively multilingual meaning representations in a cost-effective way. It includes features for multilingual & multimodal machine translation enabled by distributed training of independent encoders and decoders connected via an inner-attention layer. 

FoTraNMT is optimized for training large models (on a sufficiently large high-performance cluster). For this, we pay special attention at 
1. distributing the modules across several processing units, and 
2. efficiently traininig the network over many translation direction.  

After training, the system can also be used in  non- resource-intensive settings, because its modular design allows each individual module to be loaded and used independently. _We plan to provide trained modules, so the community can benefit from this feature._


## Requirements & Instalation 
You need to clone this repo
```bash
git clone --branch att-brg https://github.com/Helsinki-NLP/OpenNMT-py.git
cd OpenNMT-py
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

First you need to have installed [`torch`](https://pytorch.org/get-started/locally/) according to your system requirements (which can be checked using `nvidia-smi`)
```bash
python3 -m pip install torch torchvision
```

After installing pytorch, you can run: 
```bash
pip install -r requirements.txt
```

# Hands-on example

The following scripts require subword-nmt and sacrebleu.
```bash
pip install subword-nmt
pip install sacrebleu


git clone --branch att-brg https://github.com/Helsinki-NLP/OpenNMT-py.git
```

Example on how to train a simple model with 2 encoder and 1 decoder.   
First, prepare the parallel data for training, validation, and testing.
```bash
cd OpenNMT-py/data 
source ./prep-data.sh
```
   
Second, let's train a model using French and German as input, and Czech as target language.

```bash
bash train_example.sh
```
It runs on cpu, and it will train a 1-layer model for 10000 training steps   
   
After the training is completed, we can evaluate the model on a reference test.
```bash
bash test_example.sh
```

## Citing this work
If you use this work, please consider citing the work it builds up on:
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

