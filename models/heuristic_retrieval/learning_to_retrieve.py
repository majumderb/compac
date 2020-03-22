from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np

from comet.matching_utils.weak_label_annotations import get_scores, get_recall_scores, process_text

import json

# load data

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for comet expansion")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
parser.add_argument("--comet_persona", action='store_true', default=False)
parser.add_argument("--history", action='store_true', default=False)
parser.add_argument("--comet_history", action='store_true', default=False)


args = parser.parse_args()

personachat = json.load(open(args.dataset_path))
print("personachat: ", personachat.keys())
valid_data = personachat['train'] ####********

correct = 0
top_3_corr = 0
total = 0
ranks = []
debug = False #True
good = 0



class LTR(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.weight_soft = nn.Parameter(torch.ones(vocab_size))
        self.soft = nn.Softmax(dim=0)
        #self.weight = soft(self.weight_soft)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.sigm = nn.Sigmoid()
        print( list(self.parameters()) )
        
    def forward(self, list_of_doc_features, list_of_candidate_features):
        print("="*33)
        scores = []
        weight = self.soft(self.weight_soft)
        for cand_feats in list_of_candidate_features:
            #print("torch.tensor(cand_feats) ", torch.sum( torch.tensor(cand_feats)))
            #print("torch.tensor(cand_feats) ", torch.tensor(cand_feats).size())
            #print("list_of_doc_features[0] ", torch.sum(torch.tensor(list_of_doc_features[0])) )
            #tmp = [ self.sigm( torch.mean(self.weight * torch.tensor(cand_feats) * torch.tensor(df) + self.bias) ) for df in list_of_doc_features ]
            tmp = [  torch.sum(weight * torch.tensor(cand_feats) * torch.tensor(df))  for df in list_of_doc_features ]
            #print("torch.stack(tmp):", torch.stack(tmp))
            cur_score = torch.max( torch.stack(tmp) )
            #print("cur_score = ", cur_score)
            scores.append(cur_score)
        assert len(scores)==20, len(scores)
        print("scores = ", scores)
        gt_score = scores[-1]
        max_incorrect_score = torch.max( torch.stack(scores[:-1]) )
        loss = torch.max(torch.tensor(0.), 10000*(max_incorrect_score - gt_score) + 0.001 )
        print("")
        if gt_score > max_incorrect_score:
            correct = 1
        else:
            correct = 0
        return {'loss':loss, 'correct':correct}
    
        
class Vocab:
    
    def __init__(self):
        self.ctr = 0
        self.w2idxctr = {}
        self.w2idx = {}
        self.idx2w = {}
        
    def update_vocab(self, lst_of_tokens):
        for t in lst_of_tokens:
            self.w2idxctr[t] =  self.w2idxctr.get(t,0) + 1
        
    def prepare(self, thresh=5):
        for t,cnt in self.w2idxctr.items():
            if cnt>thresh:
                self.w2idx[t] = self.ctr
                self.idx2w[self.ctr] = t
                self.ctr+=1
                
    def get_feats(self, lst_of_tokens, norm=True):
        #print("get_feats : lst_of_tokens = ", lst_of_tokens)
        ret = np.zeros( self.ctr, dtype=np.float32 )
        ctr = 0
        for t in lst_of_tokens:
            if t in self.w2idx:
                ctr += 1
                ret[self.w2idx[t]] = 1.0
        if norm:
            if np.sum(ret) > 0:
                ret = ret / np.sum(ret)
        #print("get_feats : ctr = ", ctr)
        #print("get_feats : ret = ", ret)
        return ret
        

class Solver:
    
    def __init__(self):
        self.vocab = Vocab()
    
    def preprocess(self, data):
        print("===== PREPROCESS ====")
        thresh=5
        for d_i, dialog in tqdm(enumerate(data), total=len(data)):
            for u_i, utterance in enumerate(dialog['utterances']):
                vals = self.get_doc(utterance, dialog)
                # {'grounding_doc':grounding_doc, 'candidate_docs':candidate_docs}
                for s in vals['grounding_doc']:
                    self.vocab.update_vocab(s)
                for s in vals['candidate_docs']:
                    self.vocab.update_vocab(s)
            if debug and d_i>0:
                thresh=0
                break
        self.vocab.prepare(thresh=thresh)
        self.model = LTR(self.vocab.ctr)
        print("self.model = ", self.model)
        
    #def get_batch(self, data, i, bsz):
    #    ret = data[i*bsz:(i+1)*bsz]
    #    ret = [self.vocab.get_feats(doc) for doc in ret]
    #    return ret
    
    #def get_num_batches(self, data, bsz):
    #    return (len(data)+bsz-1)//bsz
    
    def train_epoch(self, data, optim):
        correct = 0.0
        total = 0.0
        for d_i, dialog in tqdm(enumerate(data), total=len(data)):
            valsd = self.get_grounding_doc(dialog)
            grounding_docvals = valsd['grounding_doc']
            #print("****** grounding_doc = ", grounding_doc)
            #assert len(grounding_doc)>0, " len(grounding_doc) found to be 0"
            if True:
                grounding_doc = []
                for v in grounding_docvals:
                    grounding_doc += v
                grounding_doc = [grounding_doc]
            else:
                grounding_doc = grounding_docvals
            grounding_doc = [self.vocab.get_feats(doc, norm=False) for doc in grounding_doc]
            for u_i, utterance in enumerate(dialog['utterances']):
                vals = self.get_candidate_docs(utterance)
                candidate_docs = vals['candidate_docs']
                candidate_docs = [self.vocab.get_feats(doc) for doc in candidate_docs]
                ret = self.model(grounding_doc, candidate_docs)
                loss = ret['loss']
                correct += ret['correct']
                total += 1              
                #loss += 0.01* ( torch.sum( self.model.weight*self.model.weight) + torch.sum(self.model.bias*self.model.bias) )
                print("loss = ", loss)
                self.model.zero_grad()
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(" =====> acc = ", 100.0*correct/total)
            if debug and d_i>3:
                break
            print(" =====> acc = ", 100.0*correct/total)
            
    def train(self, data, vals):
        #print("self.model.weight.data = ", list(self.model.weight.data))
        epochs = vals['epochs']
        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(epochs):
            self.train_epoch(data, optim)
            if debug and epoch>5:
                break
        print("self.model.weight_soft.data = ", list(self.model.weight_soft.data))
        #print("self.model.bias.data = ", list(self.model.bias.data))
        #for i,wt in enumerate(self.model.weight.data):
            
        # create model, optims
        # iterate over epochs
        # update
                    
    def get_doc(self, utterance, dialog):
        tmp = self.get_grounding_doc(dialog)
        tmp.update(self.get_candidate_docs(utterance))
        return tmp
        
    def get_grounding_doc(self, dialog):
         
        grounding_doc = []
        # add personality
        grounding_doc += dialog["personality"]
        persona_len = len(dialog["personality"])

        if args.comet_persona:
            # add comet expansions for personality
            comet_annotations = dialog["coment_annotation"]
            sent_beams_persona = []
            for j_s, sent in enumerate(comet_annotations):
                for effect_name, effect in sent['comet'].items():
                    sent_beams_persona += effect['beams'][:args.num_beams]
            grounding_doc += sent_beams_persona

        if args.history:
            grounding_doc += utterance['history'][-(2*args.max_history+1):]

        if args.comet_history:
            comet_history = dialog["history_comet_annotation"][:(2*u_i + 1)][-(2*args.max_history+1):]
            sent_beams_history = []
            for j_s, sent in enumerate(comet_history):
                for effect_name, effect in sent['comet'].items():
                        sent_beams_history += effect['beams'][:args.num_beams]
            grounding_doc += sent_beams_history

        grounding_doc = [process_text(s, typ='unigram') for s in grounding_doc] ### list of vector features
        ### can make it consistent -- persona + last_two + expanded_persona + expanded_lasttwo
        ### then can be viewed as vector of vector 
        return {'grounding_doc':grounding_doc}
        
    def get_candidate_docs(self, utterance):
        candidate_docs = [process_text(s, typ='unigram') for s in utterance['candidates']]
        return {'candidate_docs':candidate_docs}

        
solver = Solver() 
solver.preprocess(data=valid_data)
solver.train(valid_data, {'epochs':11})

        
        
        
# print('Total {} utterances retrieved'.format(total))
# print('Accuracy: {}'.format(correct / total))

# print('Top-3 Accuracy: {}'.format(top_3_corr / total))

# mrr = sum([1/r for r in ranks])/ len(ranks)
# print('MRR: {}'.format(mrr))

# print(good)



# '''
# python3 learning_to_retrieve.py --dataset_path data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_history_preprocessed_train.json 
# python learning_to_retrieve.py --dataset_path data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_history_preprocessed_train.json --comet_persona
# '''

        
    