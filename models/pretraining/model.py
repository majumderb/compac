from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reinforce_model.prior_posterior_models import PriorBoWModel, PriorRobertaModel

'''
Posterior Pretraining:
- given correct_candiate[,history], score each persona sentence
- for each candidate, inp=[history,candidate]; output=scores_over_persona
- collect scores for all candidates

Prior pretraining:
- given history, pick persona which is closest to the correct candidate
- compute following scores for correct candidate, all persona: [persona]
- output logits from prior should be guided towards above scores
'''

'''
1. Roberta based
2. Simpler weighted ngrams based
'''


class PriorPretrainingModel(nn.Module):
    def __init__(self,
                 args
                 ):
        super().__init__()

        self.args = args
        if args.prior_model == 'bow':
            self.prior_model = PriorBoWModel(args)
        elif args.prior_model == 'roberta':
            self.prior_model = PriorRobertaModel(args)
        else:
            raise Exception('Invalid prior model')
        self.criterion_mc = torch.nn.MSELoss() #reduction='none')

    def get_score_z_given_H(self, persona, history):
        return self.prior_model.get_prob_z_given_H(persona=persona, history=history)

    def get_score_z_given_goldcandidate(self, persona, response):
        return self.prior_model.get_prob_z_given_H(persona=persona, history=response)


    def forward(
        self,
        input_ids,
        token_type_ids,
        persona=None,
        history=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None):
        '''
        persona: B x P x T
        input_ids: B x P x C x T
        mc_token_ids:
        lm_labels: B x P x C x T
        mc_labels: B
        token_type_ids: B x P x C x T 
        '''
        mc_token_ids_gt = mc_token_ids[:,:,0]
        z_given_h = self.get_score_z_given_H(persona, history) # B x P - unnormalized scores
        desired_z = self.get_score_z_given_H(persona, mc_token_ids_gt) # B x P - unnormalized scores
        loss_prior_pretraining = self.criterion_mc(z_given_h, desired_z)
        return loss_prior_pretraining
