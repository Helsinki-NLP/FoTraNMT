# Use on Mahti 

It is different to use our system on Mahti since we the available modules do not cover our requirements.
I am documenting my efforts on getting it running. Is successful, this file will serve to replicate it easily.

Need to load the modules
```
module load pytorch/1.8
module load git <- not sure I need it
```

This already inclued a lot of the needed requirement - obviously most of them with different versions. The ones not loaded, should be installed at a user level:

```
python -m pip install --user configargparse subword-nmt sacrebleu ipdb 
```
and the optional reuirements
```
python -m pip install --user pyrouge pyonmttok opencv-python flask
```

