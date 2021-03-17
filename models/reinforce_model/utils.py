from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"


def preprocess(effect_type, beam):
    if effect_type == 'xAttr':
        beam = 'i am ' + beam
    elif effect_type == 'xEffect':
        if 'personx' not in beam:
            beam = 'i ' + beam
        beam = beam.replace('personx', 'i')
    elif effect_type == 'xIntent':
        beam = 'i want ' + beam
    elif effect_type == 'xNeed':
        beam = 'i need ' + beam
    elif effect_type == 'xReact':
        beam = 'i feel ' + beam
    elif effect_type == 'xWant':
        beam = 'i want ' + beam
    else:
        beam = beam
    
    return beam + ' .'


def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    print("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path
    dataset_dir = os.path.dirname(os.path.realpath(dataset_path))
    dataset_cache = os.path.join(dataset_dir, dataset_cache + '_cache_' + type(tokenizer).__name__)
    print('Looking for cache at {}'.format(dataset_cache))
    if dataset_cache and os.path.isfile(dataset_cache):
        print("Load tokenized dataset from cache at {}".format(dataset_cache))
        dataset = torch.load(dataset_cache)
    else:
        print("Loading dataset from {}".format(dataset_path))
        with open(dataset_path, "r+", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        print("Tokenize and encode the dataset")
        start = datetime.now()
        def tokenize(obj):
            if isinstance(obj, float) or isinstance(obj, int):
                return obj
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) if n != "comet_key" else (n, o) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
        print('{} - Cached dataset at {}'.format(datetime.now() - start, dataset_cache))
    return dataset

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str, exp_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name + '_' + exp_name)
    return logdir


"""
def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)
"""
