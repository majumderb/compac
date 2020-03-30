from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorModel(nn.Module):

    def __init__(self,
                 args):
        super().__init__()
        self.args = args
        self.uniform_prior = args.uniform_prior
        if not self.uniform_prior:
            self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', output_hidden_states=True)


    def get_prob_z_given_H(self, persona, history):
        '''
        persona: B x P x T
        H: B x T

        We take the pooled output from Roberta; which uses same tokenization as GPT2
        TODO: Add <s> token at the beginning which will act as [CLS] token in BERT
        '''

        if self.uniform_prior:
            num_persona = persona.shape[1]
            prob_z_given_H = torch.ones([persona.shape[0], persona.shape[1]]) / num_persona  # B x P

            return prob_z_given_H.to(self.args.device)

        else:
            history_encodings = self.roberta_model(history)[1][-1][:, 0, :]  # B x 764
            history_encodings = history_encodings.unsqueeze(1).repeat(1, persona.shape[1], 1)  # B x P x 764

            persona_encodings = []
            for i in range(persona.shape[1]):
                persona_enc = self.roberta_model(persona[:, i, ...])[1][-1][:, 0, :]  # B x 764
                persona_encodings.append(persona_enc)

            persona_encodings = torch.stack(persona_encodings, axis=1)

            norms = -1.0 * torch.norm(history_encodings - persona_encodings, 2, dim=-1)
            prob_z_given_H = F.softmax(norms, dim=-1)

            return prob_z_given_H  # B x P

    def sample(self, dist_over_z):
        '''

        :param dist_over_z: B,prior_size
        :return: action, logprob of chosen action
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        action_idx = dist.sample()  # B
        return action_idx, dist.log_prob(action_idx)  #B;B

