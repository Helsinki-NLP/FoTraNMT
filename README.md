# About this branch
This is an OpenNMT-py fork developed by the NLP group at University of Helsinki.
In this branch we developed features for multilingual & multimodal machine translation enabled by parallel training of independent encoders and decoders connected via an inner-attention layer. 

More detail on the arquitecture can be fount in [this article](https://www.aclweb.org/anthology/2020.cl-2.5)
and on the introduction of audio processing features in [this other article](https://www.aclweb.org/anthology/2020.iwslt-1.10). See master branch of this fork for documentation of OpenNMT-py functionalities or [their offical repo](https://github.com/OpenNMT/OpenNMT-py) for more updated information.

### Requirements & Instalation 
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

# References
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
    url = "https://www.aclweb.org/anthology/2020.cl-2.5",
    doi = "10.1162/coli_a_00377",
    pages = "387--424"
```    
[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)
```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
[Audio Precessing Features](https://www.aclweb.org/anthology/2020.iwslt-1.10)
```latex
@inproceedings{vazquez-etal-2020-university,
    title = "The {U}niversity of {H}elsinki Submission to the {IWSLT}2020 Offline {S}peech{T}ranslation Task",
    author = {V{\'a}zquez, Ra{\'u}l  and
      Aulamo, Mikko  and
      Sulubacak, Umut  and
      Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the 17th International Conference on Spoken Language Translation",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.iwslt-1.10",
    doi = "10.18653/v1/2020.iwslt-1.10",
    pages = "95--102"
}
```
