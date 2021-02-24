# Persona Expansion with Commonsense in Dialog
Personalized Dialog Generation with Commonsense

# Data

Huggingface cleaned dataset

`wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json`

(Optional) Personachat Download:
http://parl.ai/downloads/personachat/personachat.tgz

## COMET-expanded PersonaChat

You can download PersonaChat dataset with added COMET expansions from here[https://drive.google.com/file/d/1tJih0IecAmP3IlP6TYvDjy3kOpIbMUIH/view?usp=sharing]  

# COMeT predictions

Clone repo from: https://github.com/atcbosselut/comet-commonsense
From repo:

Then run the setup scripts to acquire the pretrained model files from OpenAI, as well as the ATOMIC and ConceptNet datasets
```
bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_conceptnet_data.sh
bash scripts/setup/get_model_files.sh
```
Make sure you have all the requirements mentioned here in README: https://github.com/atcbosselut/comet-commonsense

Make preprocessed data loader for ATOMIC and CONCEPTNETS

```
python scripts/data/make_atomic_data_loader.py
python scripts/data/make_conceptnet_data_loader.pypython scripts/data/make_atomic_data_loader.py
```

Pretrined models can be downloaded from here: `https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB`

Unzip the file: `tar -xvzf pretrained_models.tar.gz`

## Interactive Mode

Play with COMeT completions here: `python scripts/interactive/atomic_single_example.py --model_file pretrained_models/atomic_pretrained_model.pickle`

Choose `all` as `effect type`. Other options as follow:
```
all - compute the output for all effect types {{oEffect, oReact, oWant, xAttr, xEffect, xIntent, xNeed, xReact, xWant}}
oEffect - generate the effect of the event on participants other than PersonX
oReact - generate the reactions of participants other than PersonX to the event
oEffect - generate what participants other than PersonX may want after the event
```
Choose `beam-5` as decoding algorithm. Other options as follow:
```
greedy
beam-# where # is the beam size
topk-# where # is k
```
We will change this code to be able take an input json and produce an output json with all expansions.
