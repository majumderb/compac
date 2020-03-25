from collections import defaultdict
from itertools import chain
from argparse import ArgumentParser

from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.discrete_choice_model.utils import get_dataset, make_logdir, preprocess
from datetime import datetime

import numpy as np
import torch
import json
import os

ROBERTA_START = 2
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
EFFECTS = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
PERSONA_MAX_LENGTH = 350

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))[:PERSONA_MAX_LENGTH]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    print('\nseq', sequence)
    print('\npersona', persona)
    instance["persona"] = [[ROBERTA_START] + p  for p in persona]
    instance["persona_length"] = [len(p) for p in instance["persona"]]
    instance["history"] = [ROBERTA_START] + history
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

    # print("Pad inputs and convert to Tensor")
    # tensor_datasets = {"train": [], "valid": []}
    # for dataset_name, dataset in datasets.items():
    #     dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
    #     for input_name in MODEL_INPUTS:
    #         tensor = torch.tensor(dataset[input_name])
    #         if input_name != "mc_labels":
    #             tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
    #         tensor_datasets[dataset_name].append(tensor)

    # print("Build train and validation dataloaders")
    # train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    # train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    # valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    # print("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    # print("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    # return train_loader, valid_loader, train_sampler, valid_sampler

class PersonaChatDataset(Dataset):
    def __init__(
            self,
            args,  # Bookkeeping
            tokenizer,
            split,
            debug_mode=False,  # Debugging
            sample=None,
            **kwargs,
    ):
        super().__init__()

        self.split = split

        personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
        print("Build inputs and labels for {}".format(split))

        self.datasets = {split: defaultdict(list)}
        # for dataset_name, dataset in personachat.items():
        dataset = personachat[split]
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and split == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        
        if args.test_run_num > 0:
            dataset = dataset[:args.test_run_num]

        for d_i, dialog in enumerate(dataset):
            persona = dialog["personality"].copy()
            if not args.no_comet_persona:
                comet_annotations = dialog["coment_annotation"]
                sent_beams = []
                for j_s, sent in enumerate(comet_annotations):
                    # logging
                    if d_i == 0 and j_s == 0:
                        print('For a sent: \n{}'.format(sent['comet']))
                    for effect_name, effect in sent['comet'].items():
                        # if effect_name in EFFECTS:
                            # logging
                            if d_i == 0 and j_s == 0:
                                print('Getting data for effect {}'.format(effect_name))
                                print('Getting {} beams'.format(len(effect['beams'][:args.num_beams])))
                            sent_beams += effect['beams'][:args.num_beams]
                if d_i == 0:
                    print('Got {} beams'.format(len(sent_beams)))        
                persona += sent_beams
            
            for perm in range(args.personality_permutations):
                if args.no_persona:
                    persona = [[]]
                for i, utterance in enumerate(dialog["utterances"]):
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        print('instance: {}'.format(instance))
                        for input_name, input_array in instance.items():
                            self.datasets[split][input_name].append(input_array)
                    self.datasets[split]["mc_labels"].append(num_candidates - 1)
                    self.datasets[split]["n_candidates"] = num_candidates
                # persona = [persona[-1]] + persona[:-1]  # permuted personalities

    def _sample(self, n=1):
        """
        For debugging purposes. Samples random turns
        """
        return [self[i] for i in np.random.choice(len(self.index), n, replace=False)]

    def __len__(self):
        return len(self.datasets[self.split])

    def __getitem__(self, index):

        item = []
        for name in self.datasets[self.split].keys():
            item.append(self.datasets[self.split][name][index])
        
        return item


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


'''
from models.discrete_choice_model.dataset import PersonaChatDataset
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
import torch
args = torch.load('/data2/bodhi/projects/persona-dialog/models/baseline_w_comet/runs/Mar03_00-35-44_deepyeti_gpt2test/model_training_args.bin')
args.dataset_cache = 'persona_comet_weak_label_preprocessed'
args.personality_permutations = 1
args.dataset_path='/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json'
args.no_comet_persona=True
dataset = PersonaChatDataset(args, tokenizer, split='train')
'''