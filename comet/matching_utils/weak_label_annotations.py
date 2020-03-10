
# coding: utf-8


DATA_FILE = '../../data/personachat_self_original_comet_scores.json'


val_only = False
DUMP_FILE = None


if val_only:
    ### -- running  with validation split only
    # DUMP_FILE = '../../data/personachat_self_original_comet_validation_scores_alignlabels.json'
    DUMP_FILE = '../../data/personachat_self_original_comet_validation_scores_alignlabels.expanded_persona.json'
    # python persona_explorations_annotations.py > persona_explorations_annotations_log_valonly
else:
    ###  -- running with entire data
    DUMP_FILE = '../../data/personachat_self_original_comet_scores_alignlabels.expanded_persona.json'
    #python persona_explorations_annotations.py > persona_explorations_annotations_lo



print("DATA_FILE = ", DATA_FILE)
print("DUMP_FILE = ", DUMP_FILE)

import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

# import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive



from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import string
from nltk.corpus import stopwords 

ps = PorterStemmer() 
stop_words = set(stopwords.words('english')) 

def process_text(s, typ='unigram', rem_stop=True, do_stem=True):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(s) 
    filtered_sentence = word_tokens
    if rem_stop:
        filtered_sentence = [w for w in filtered_sentence if not w in stop_words] 
    if do_stem:
        filtered_sentence = [ps.stem(w) for w in filtered_sentence] 
    if typ=='bigram':
        filtered_sentence = [ '_'.join(filtered_sentence[i:i+2]) for i in range(len(filtered_sentence)-1) ] 
    return filtered_sentence
    


def get_scores(s1, s2, idx=None): # both processed
    s1 =set(s1)
    s2 = set(s2)
    num = len(s1.intersection(s2))
    den = len(s1.union(s2))    
    if den>0.0:
        score = num*1.0/den
    else:
        score = 0.0
    return {'score':score, 'idx':idx, 's1':list(s1), 's2':list(s2)}


def get_recall_scores(s1, s2, idx=None): # both processed
    s1 =set(s1)
    s2 = set(s2)
    num = len(s1.intersection(s2))
    den = len(s1)  # s1 is the sentence of interest being matched over s2 belonging to a set   
    if den>0.0:
        score = num*1.0/den
    else:
        score = 0.0
    return {'score':score, 'idx':idx, 's1':list(s1), 's2':list(s2)}
    
    
    
def match_sentence(sentence, reference, k=3, score_thresh=0.1):
    ret = [ get_scores(sentence, ref_sentence, i) for i,ref_sentence in enumerate(reference) ]
    ret = sorted( ret, key=lambda x:-x['score'] )
    ret = [vals for vals in ret if vals['score']>=score_thresh]
    return ret[:k]

def match_sentence(sentence, reference, k=3, score_thresh=0.1, use_recall_scores = False):
    if use_recall_scores:
        ret = [ get_recall_scores(sentence, ref_sentence, i) for i,ref_sentence in enumerate(reference) ]
    else:
        ret = [ get_scores(sentence, ref_sentence, i) for i,ref_sentence in enumerate(reference) ]
    ret = sorted( ret, key=lambda x:-x['score'] )
    ret = [vals for vals in ret if vals['score']>=score_thresh]
    return ret[:k]
    
def heuristic_matching(item, typ='unigram', rem_stop=True, persona_idx=None, k=3, score_thresh=0.1, alternate=True):
    
    utterances = item['utterances']
    history = utterances[-1]['history']
    personality = item['personality']
    coment_annotation = item['coment_annotation']
    
    all_dialog = history + [utterances[-1]['candidates'][-1]]
    
    personality_processed = [ process_text(p,typ=typ,rem_stop=rem_stop) for p in personality]
        
    weak_label_persona = []
    for h,sent_h in enumerate(all_dialog):
        sent_h_processed = process_text(sent_h)
        label_persona = match_sentence(sent_h_processed, personality_processed,k=k,score_thresh=score_thresh)
        if alternate and h%2==0:
            label_persona = []
        else:
            label_persona = match_sentence(sent_h_processed, personality_processed,k=k,score_thresh=score_thresh)
        cur = {'label_persona':label_persona, 'sentence':sent_h}
        weak_label_persona.append(cur)
    
    ret = weak_label_persona
    
    return ret
    
def heuristic_matching_comet(item, typ='unigram', 
                       rem_stop=True, 
                       persona_idx=None, 
                       max_matches=3, 
                       score_thresh=0.1, 
                       alternate=True, 
                       use_recall_scores=False):
    
    utterances = item['utterances']
    history = utterances[-1]['history']
    personality = item['personality']
    coment_annotation = item['coment_annotation']
    
    all_dialog = history + [utterances[-1]['candidates'][-1]]
    
    personality_processed = [ process_text(p,typ=typ,rem_stop=rem_stop) for p in personality]
    
    weak_label_comet_persona = []
    for h,sent_h in enumerate(all_dialog):
        #print("="*22)
        tmp_h = []
        sent_h_processed = process_text(sent_h)
        #print("sent_h_processed = ", sent_h_processed)
        if alternate and h%2==0:
            pass
        else:
            for i,coment_annotation_i in enumerate(coment_annotation):
                coment_annotation_i = coment_annotation_i['comet'] # dictionary
                for j,comet_keyj in enumerate(coment_annotation_i.keys()):
                    for k,beam in enumerate(coment_annotation_i[comet_keyj]['beams']):
                        ijk_processed = process_text(beam)
                        #print("ijk_processed = ", ijk_processed)
                        tmp_ijk = match_sentence( sent_h_processed, 
                                                 [ijk_processed], 
                                                 use_recall_scores = use_recall_scores,
                                                score_thresh=score_thresh)
                        if len(tmp_ijk) > 0:
                            tmp_ijk = tmp_ijk[0] # there is only one entry
                            score = tmp_ijk['score']
                            tmp_h.append( [ { 'persona_sent_id':i, 
                                             'comet_key':comet_keyj, 
                                             'beam_id':k, 
                                             'entry':beam }, 
                                           score ] )
            #print(" max_matches = ", max_matches)
            tmp_h = sorted( tmp_h, key=lambda x:-x[1] )[:max_matches]
        cur = {'label_persona':tmp_h, 'sentence':sent_h}
        weak_label_comet_persona.append(cur)
    
    ret = weak_label_comet_persona
    
    return ret
        

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



# opt, state_dict = interactive.load_model_file(args.model_file)

# data_loader, text_encoder = interactive.load_data("atomic", opt)

# n_ctx = data_loader.max_event + data_loader.max_effect
# n_vocab = len(text_encoder.encoder) + n_ctx
# model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

# if args.device != "cpu":
#     cfg.device = int(args.device)
#     cfg.do_gpu = True
#     torch.cuda.set_device(cfg.device)
#     model.cuda(cfg.device)
# else:
#     cfg.device = "cpu"


# # In[23]:


# sampling_algorithm = args.sampling_algorithm

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
                    row['weak_labels'] = heuristic_matching(row, k=3, score_thresh=0.15)
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

print(solver.data.keys())
tmp = heuristic_matching(solver.data['valid'][0], k=3, score_thresh=0.15)  
# tmp = json.dumps(tmp, indent=4)
for j in range(len(tmp)):
    print(j, tmp[j])
    
# ***** NEW
weak_label_comet_persona = heuristic_matching_comet(solver.data['valid'][0], max_matches=5, score_thresh=0.1)  
for j in range(len(weak_label_comet_persona)):
    print(j, weak_label_comet_persona[j])   

solver.process_all(DUMP_FILE, debug=False, val_only=val_only)





