from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.uniform_prior = args.uniform_prior
        if not self.uniform_prior:
            self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', output_hidden_states=True)
        # self.gpt2_model = generator_class.from_pretrained(args.model_checkpoint)
        # self.criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.criterion_mc = torch.nn.MSELoss() #reduction='none')

    def get_score_z_given_H(self, persona, history):
        '''
        persona: B x P x T
        H: B x T

        We take the pooled output from Roberta; which uses same tokenization as GPT2
        TODO: Add <s> token at the beginning which will act as [CLS] token in BERT
        '''
        history_encodings = self.roberta_model(history)[1][-1][:, 0, :] # B x 764
        history_encodings = history_encodings.unsqueeze(1).repeat(1, persona.shape[1], 1) # B x P x 764
        persona_encodings = []
        for i in range(persona.shape[1]):
            persona_enc = self.roberta_model(persona[:, i, ...])[1][-1][:, 0, :] # B x 764
            persona_encodings.append(persona_enc)
        persona_encodings = torch.stack(persona_encodings, axis=1)
        norms = -1.0 * torch.norm(history_encodings-persona_encodings, 2, dim=-1)
        prob_z_given_H = F.softmax(norms, dim=-1)
        return prob_z_given_H #  B x P

    def get_score_z_given_goldcandidate(self, persona, response):
        '''
        persona: B x P x T
        H: B x T

        We take the pooled output from Roberta; which uses same tokenization as GPT2
        TODO: Add <s> token at the beginning which will act as [CLS] token in BERT
        '''
        response_encodings = self.roberta_model(response)[1][-1][:, 0, :] # B x 764
        response_encodings = response_encodings.unsqueeze(1).repeat(1, persona.shape[1], 1) # B x P x 764
        persona_encodings = []
        for i in range(persona.shape[1]):
            persona_enc = self.roberta_model(persona[:, i, ...])[1][-1][:, 0, :] # B x 764
            persona_encodings.append(persona_enc)
        persona_encodings = torch.stack(persona_encodings, axis=1)
        norms = -1.0 * torch.norm(response_encodings-persona_encodings, 2, dim=-1)
        prob_z_given_H = F.softmax(norms, dim=-1)
        return prob_z_given_H #  B x P


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

        print("mc_token_ids : ", mc_token_ids.size())
        print("persona : ", persona.size())
        print("history : ", history.size())
        print("mc_token_ids = ", mc_token_ids)
        print("mc_labels : ", mc_labels.size())
        print("mc_labels = ", mc_labels)
        mc_token_ids_gt = mc_token_ids[:,:,0]
        print("mc_token_ids_gt: ", mc_token_ids_gt.size())
        z_given_h = self.get_score_z_given_H(persona, history) # B x P - unnormalized scores
        print("****** z_given_h = ", z_given_h)
        desired_z = self.get_score_z_given_H(persona, mc_token_ids_gt) # B x P - unnormalized scores
        print("****** desired_z = ", desired_z)
        loss_prior_pretraining = self.criterion_mc(z_given_h, desired_z)
        print("****** loss_prior_pretraining = ", loss_prior_pretraining)
        # mc_token_ids = mc_token_ids[:, i, ...].contiguous()
        return loss_prior_pretraining
