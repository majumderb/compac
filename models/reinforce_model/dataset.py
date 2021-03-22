from collections import defaultdict
from itertools import chain
from argparse import ArgumentParser

from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.reinforce_model.utils import get_dataset, make_logdir, preprocess
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import json
import os

ROBERTA_START = 2
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = [
    "input_ids",
    "token_type_ids",
    "mc_token_ids",
    "lm_labels",
    "mc_labels",
    "persona",
    "history",
    "effects",
]
EFFECTS = {
    'oEffect': 1,
    'oReact': 2,
    'oWant': 3,
    'xAttr': 4,
    'xEffect': 5,
    'xIntent': 6,
    'xNeed': 7,
    'xReact': 8,
    'xWant': 9,
    'Persona': 10,
    'Null': 0,
    }

MAX_NUM_PERSONA = 5
MAX_NUM_COMET_PERSONA = 250


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance


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
        [self.pad_id] = tokenizer.convert_tokens_to_ids(["<pad>"])
        self.split = split
        self.length = 0

        personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
        print("Build inputs and labels for {}".format(split))

        self.dataset = {n: [] for n in MODEL_INPUTS}
        # for dataset_name, dataset in personachat.items():
        personachat_split = personachat[split]
        self.num_candidates = len(personachat_split[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0: #and split == 'train':
            self.num_candidates = min(args.num_candidates, self.num_candidates)
        
        if args.test_run_num > 0:
            personachat_split = personachat_split[:args.test_run_num]
        
        print('Restricted to {} dialogs'.format(len(personachat_split)))

        for d_i, dialog in tqdm(enumerate(personachat_split), total=len(personachat_split)):
            effects = []
            persona = dialog["personality"].copy()
            effects += [EFFECTS['Persona']]*len(persona)
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
                            effects += [EFFECTS[effect_name]] * args.num_beams
                if d_i == 0:
                    print('Got {} beams'.format(len(sent_beams)))
                    print('Got {} effects'.format(len(effects)))        
                persona += sent_beams
            assert len(persona) == len(effects)
            for perm in range(args.personality_permutations):
                if args.no_persona:
                    persona = [[]]
                for i, utterance in enumerate(dialog["utterances"]):
                    history = utterance["history"][-(2*args.max_history+1):]
                    sample = {
                        "persona": [[ROBERTA_START] + p for p in persona],
                        "history": [ROBERTA_START] + list(chain(*history)),
                        "effects": effects,
                    }
                    for name in self.dataset.keys():
                        if name not in sample:
                            sample[name] = []
                    for persona_sample in persona:
                        for j, candidate in enumerate(utterance["candidates"][-self.num_candidates:]):
                            instance = build_input_from_segments(
                                [persona_sample], history, candidate, tokenizer, j == self.num_candidates-1)
                            for input_name, input_array in instance.items():
                                sample[input_name].append(input_array)
                        
                        sample["mc_labels"].append(self.num_candidates - 1)
                    for name, value in sample.items():
                        self.dataset[name].append(value)
                    self.length += 1 
                # persona = [persona[-1]] + persona[:-1]  # permuted personalities

    def _sample(self, n=1):
        """
        For debugging purposes. Samples random turns
        """
        return [self[i] for i in np.random.choice(self.length, n, replace=False)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        '''
        input_ids
        token_type_ids
        mc_token_ids
        lm_labels
        persona
        history
        mc_labels
        '''
        sample = {}
        for name in self.dataset.keys():
            sample[name] = self.dataset[name][index]
        return sample

    def collate_dialog(self, batch):
        '''
        Padding and Collating
        '''
        max_num_persona = max(len(b['persona']) for b in batch)
        for b_i, b in enumerate(batch):
            b['persona'] += [[] for _ in range(max_num_persona -len(b['persona']))]
        max_seq_len = max(len(c) for b in batch for c in b['input_ids'])
        max_persona_len = max(len(p) for b in batch for p in b['persona'])
        max_history_len = max(len(b['history']) for b in batch)
        padded_batch = {}
        for name in batch[0].keys():
            if name in ['input_ids', 'token_type_ids']:
                padded = torch.LongTensor(
                    [[c + [self.pad_id]*(max_seq_len - len(c)) for c in sample[name]] for sample in batch])
                padded_batch[name] = padded.view((-1, max_num_persona, self.num_candidates) + padded.shape[2:])
            elif name == 'persona': 
                padded_batch[name] = torch.LongTensor(
                    [[p + [self.pad_id]*(max_persona_len - len(p)) for p in sample['persona']] for sample in batch])
            elif name == 'history':
                padded_batch[name] = torch.LongTensor(
                    [sample[name] + [self.pad_id]*(max_history_len - len(sample[name])) for sample in batch])
            elif name == "lm_labels":
                padded = torch.LongTensor(
                    [[c + [-100]*(max_seq_len - len(c)) for c in sample[name]] for sample in batch])
                padded_batch[name] = padded.view((-1, max_num_persona, self.num_candidates) + padded.shape[2:])
            elif name == "mc_token_ids":
                padded_batch[name] = torch.LongTensor([sample[name] for sample in batch])\
                    .view((-1, max_num_persona, self.num_candidates))
            elif name in ["mc_labels", "effects"]:
                padded_batch[name] = torch.LongTensor([sample[name] for sample in batch])
            else:
                assert False, f"Unexpected batch element with key '{name}'"
        # print("PersonaChatDataset.collate_dialog:")
        # for k, v in padded_batch.items():
        #     print(f"{k}.shape:", v.shape)
        return padded_batch        


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
from models.reinforce_model.dataset import PersonaChatDataset, ATTR_TO_SPECIAL_TOKEN, collate_dialog, build_input_from_segments
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
import torch
args = torch.load('/data2/bodhi/projects/persona-dialog/models/baseline_w_comet/runs/Mar03_00-35-44_deepyeti_gpt2test/model_training_args.bin')
args.dataset_cache = 'persona_comet_weak_label_preprocessed'
args.num_candidates = 1
args.personality_permutations = 1
args.test_run_num = -1
args.dataset_path='/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json'
args.no_comet_persona=True
dataset = PersonaChatDataset(args, tokenizer, split='train')
batch = dataset._sample(2)
padded_input_ids, padded_token_type_ids, padded_lm_labels, mc_token_ids, mc_labels, padded_persona, padded_history, padded_effects = collate_dialog(batch)
'''
