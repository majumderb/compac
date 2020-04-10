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
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
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

PERSONA_MAX_LENGTH = 50
MAX_NUM_PERSONA = 5
MAX_NUM_COMET_PERSONA = 300

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    # print('\nseq', sequence)
    # print('\npersona', persona)
    # print('\nhistory', history)
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
        self.length = 0

        personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
        print("Build inputs and labels for {}".format(split))

        self.dataset = defaultdict(list)
        # for dataset_name, dataset in personachat.items():
        personachat_split = personachat[split]
        num_candidates = len(personachat_split[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0: #and split == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        
        if args.test_run_num > 0:
            personachat_split = personachat_split[:args.test_run_num]
        
        print('Restricted to {} dialogs'.format(len(personachat_split)))

        effects = []
        for d_i, dialog in tqdm(enumerate(personachat_split), total=len(personachat_split)):
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
            
            for perm in range(args.personality_permutations):
                if args.no_persona:
                    persona = [[]]
                else:
                    if args.no_comet_persona:
                        persona = persona + [[0]]*(MAX_NUM_PERSONA - len(persona))
                        effects = effects + [0]*(MAX_NUM_PERSONA - len(effects))
                    else:    
                        persona = persona + [[0]]*(MAX_NUM_COMET_PERSONA - len(persona))
                        effects = effects + [0]*(MAX_NUM_COMET_PERSONA - len(effects))
                for i, utterance in enumerate(dialog["utterances"]):
                    history = utterance["history"][-(2*args.max_history+1):]
                    for persona_sample in persona:
                        for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                            lm_labels = bool(j == num_candidates-1)
                            instance = build_input_from_segments([persona_sample], history, candidate, tokenizer, lm_labels)
                            # print('instance: {}'.format(instance))
                            for input_name, input_array in instance.items():
                                self.dataset[input_name].append(input_array)
                        
                        self.dataset["mc_labels"].append(num_candidates - 1)

                    self.dataset["persona"].append([[ROBERTA_START] + p  for p in persona])
                    self.dataset["history"].append([ROBERTA_START] + list(chain(*history)))
                    history_folded = kwargs.get('history_folded', False)
                    if history_folded:
                        self.dataset["history_folded"].append(history)
                    self.dataset["n_candidates"] = num_candidates
                    print(persona)
                    print(effects)
                    self.dataset["effects"] = effects 

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
        n_candidates
        '''

        multiplier = self.dataset['n_candidates'] * MAX_NUM_PERSONA
        items = []
        for name in self.dataset.keys():
            if name not in ['n_candidates', 'mc_labels', 'persona', 'history', 'history_folded']:
                item = [self.dataset[name][index*multiplier:(index+1)*multiplier]]
                items.append(item)
            elif name  == 'mc_labels':
                items.append(self.dataset[name][index*MAX_NUM_PERSONA:(index+1)*MAX_NUM_PERSONA])
            elif name in ['persona', 'history', 'history_folded', 'effects']:
                items.append(self.dataset[name][index])
            elif name == 'n_candidates':
                items.append(self.dataset[name])
        
        if 'history_folded' in self.dataset.keys():
            input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels, persona, history, history_folded, n_candidates, effects = items
            return input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels, persona, history, history_folded, n_candidates, effects
        else:
            input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels, persona, history, n_candidates, effects = items
            return input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels, persona, history, n_candidates, effects

def collate_dialog(batch):
    '''
    Padding and Collating
    '''
    input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels, persona, history, n_candidates, effects = zip(*batch)
    n_candidates = n_candidates[0]

    max_seq_len = 0
    for b in input_ids:
        for c in b[0]:
            max_seq_len = max(max_seq_len, len(c))

    padded_input_ids = torch.LongTensor([
        [c + [0]*(max_seq_len - len(c)) for c in seq[0]]
        for seq in input_ids])
    padded_input_ids = padded_input_ids.view((-1, MAX_NUM_PERSONA, n_candidates) + padded_input_ids.shape[2:])

    padded_token_type_ids = torch.LongTensor([
        [c + [0]*(max_seq_len - len(c)) for c in seq[0]]
        for seq in token_type_ids])
    padded_token_type_ids = padded_token_type_ids.view((-1, MAX_NUM_PERSONA, n_candidates) + padded_token_type_ids.shape[2:])
    
    padded_lm_labels = torch.LongTensor([
        [c + [-100]*(max_seq_len - len(c)) for c in seq[0]]
        for seq in lm_labels])
    padded_lm_labels = padded_lm_labels.view((-1, MAX_NUM_PERSONA, n_candidates) + padded_lm_labels.shape[2:])

    mc_token_ids = torch.LongTensor(mc_token_ids).squeeze(1)
    mc_token_ids = mc_token_ids.view((-1, MAX_NUM_PERSONA, n_candidates))
    mc_labels = torch.LongTensor(mc_labels)

    # persona
    max_persona_len = 0
    for b in persona:
        for p in b:
            max_persona_len = max(max_persona_len, len(p))

    padded_persona = torch.LongTensor([[p + [0]*(max_persona_len - len(p)) for p in b] for b in persona])

    # history
    max_history_len = 0
    for b in history:
        max_history_len = max(max_history_len, len(b))
    padded_history = torch.LongTensor([b + [0]*(max_history_len - len(b)) for b in history])

    padded_effects = torch.LongTensor(effects)

    return padded_input_ids, padded_token_type_ids, padded_lm_labels, \
        mc_token_ids, mc_labels, padded_persona, padded_history, padded_effects


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
padded_input_ids, padded_token_type_ids, padded_lm_labels, mc_token_ids, mc_labels, padded_persona, padded_history = collate_dialog(batch)
'''