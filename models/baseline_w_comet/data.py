from collections import defaultdict
from itertools import chain
from argparse import ArgumentParser

from torch.utils.data import DataLoader, TensorDataset
from utils import get_dataset, make_logdir, preprocess
from datetime import datetime

import torch
import json
import os

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
EFFECTS = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    print("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        
        if args.test_run_num > 0:
            dataset = dataset[:args.test_run_num]

        i = 0 # for printing
        for dialog in dataset:
            persona = dialog["personality"].copy()
            comet_annotations = dialog["coment_annotation"]
            for sent in comet_annotations:
                sent_beams = []
                for effect_name, effect in sent['comet'].items():
                    if effect_name in EFFECTS:
                        if i == 0:
                            print('Getting data for effect {}'.format(effect_name))
                            print('Getting {} beams'.format(len(effect['beams'][:args.num_beams])))
                        sent_beams += effect['beams'][:args.num_beams]
                i = 1        
            persona += sent_beams

            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    print("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    print("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    print("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def preprocess_comet_dataset(dataset_path):
    with open(dataset_path, "r+", encoding="utf-8") as f:
        dataset = json.loads(f.read())
    
    for _, split in dataset.items():
        for dialog in split:
            comet_annotations = dialog['coment_annotation']
            for s in comet_annotations:
                comet = s['comet']
                for k, v in comet.items():
                    for i in range(len(v['beams'])):
                        v['beams'][i] = preprocess(k, v['beams'][i])
    
    return dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    args = parser.parse_args()

    print('Starting preprocessing')
    start = datetime.now()
    dataset = preprocess_comet_dataset(args.dataset_path)
    print('{} - Finished preprocessing.'.format(datetime.now() - start))

    save_dir = os.path.dirname(os.path.realpath(args.dataset_path))
    orig_filename = os.path.basename(args.dataset_path)
    save_filename = orig_filename[:-5] + '_preprocessed.json'


    with open(os.path.join(save_dir, save_filename), 'w') as outfile:
        json.dump(dataset, outfile)
    
    print('File saved.')
