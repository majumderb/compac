
# coding: utf-8


DATA_FILE = '/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json'

val_only = True
DUMP_FILE = '../../data/personachat_self_original_comet_validation.json'


if val_only:
    ### -- running  with validation split only
    DUMP_FILE = '/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_history_preprocessed_validation.json'
    # python persona_explorations_annotations.py > persona_explorations_annotations_log_valonly
else:
    ###  -- running with entire data
    DUMP_FILE = '../../data/personachat_self_original_comet_scores.json'
    #python persona_explorations_annotations.py > persona_explorations_annotations_lo



print("DUMP_FILE = ", DUMP_FILE)

import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

# import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


# In[9]:


class Args:
    def __init__(self, 
                 device = 'cpu',
                 model_file="pretrained_models/atomic_pretrained_model.pickle",
                 sampling_algorithm = 'beam-5'
                ):
        self.device = device
        self.model_file = model_file
        self.sampling_algorithm = sampling_algorithm


# In[10]:


args = Args()


# In[11]:



opt, state_dict = interactive.load_model_file(args.model_file)

data_loader, text_encoder = interactive.load_data("atomic", opt)

n_ctx = data_loader.max_event + data_loader.max_effect
n_vocab = len(text_encoder.encoder) + n_ctx
model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

if args.device != "cpu":
    cfg.device = int(args.device)
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
else:
    cfg.device = "cpu"


# In[23]:


sampling_algorithm = args.sampling_algorithm

def fnc(persona, debug=False): # list of sentences

    ret = []
    for sent in persona:
        
        cur_sent = {'sent':sent}
        if debug:
            print()
            print("-x"*33)
            print("=====>>>> sent = ", sent)

        input_event = sent
        category = "all"
        
        if debug:
            print()
            print("category = ", category)

        sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
        outputs = interactive.get_atomic_sequence(
            input_event, model, sampler, data_loader, text_encoder, category)
        # print("outputs = ", outputs)
        cur_sent['comet']  = outputs
        ret.append(cur_sent)

    return ret


# In[14]:


import json
import random

class AnnotatePersonaChat():
    
    def __init__(self):
        with open(DATA_FILE, "r", encoding="utf-8") as read_file:
            self.data = json.load(read_file)

        print('Read {} training examples and {} validation examples'.format(
            len(self.data['train']), len(self.data['valid'])
        ))

    def get_sample(self, split='train'):
        split = self.data[split]
        index = random.randint(0, len(split))
        sample = split[index]

        persona = sample['personality']
        utterances = sample['utterances']

        conversation = utterances[-1]['history'] + [utterances[-1]['candidates'][-1]]

        print('PERSONA {}\n{}'.format(
            '='*33, '\n'.join(persona))
        )
        print('CONVERSATION {}\n- {}'.format(
            '='*33, '\n- '.join(conversation))
        )
        return persona

        
    def process_all(self, dump_fname, debug=False, val_only=False):
        self.annotated_data = {}
        for split,data in self.data.items():
            if val_only and split=='train':
                continue
            annotated_data = []
            miss_cnt = 0
            print("******* split =",split, " || data: ", len(data) )
            
            for j,row in enumerate(data):
                try:
                    utterances = sample['utterances']
                    history = utterances[-1]['history'] + [utterances[-1]['candidates'][-1]]
                    row['history_comet_annotation'] = fnc(history)
                    annotated_data.append(row)
                except:
                    miss_cnt += 1
                    continue
                if debug:
                    break
                print("******* split =",split, " j = ", j )
                    
            self.annotated_data[split] = annotated_data
            print("******* split =",split, " || annotated_data: ", len(annotated_data), " || miss_cnt=",miss_cnt )
        json.dump(self.annotated_data, open(dump_fname,'w'))


solver = AnnotatePersonaChat()


persona_sample = solver.get_sample('train')

print( fnc(persona_sample) )



solver.process_all(DUMP_FILE, debug=False, val_only=val_only)

