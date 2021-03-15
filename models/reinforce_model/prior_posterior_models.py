from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reinforce_model.dataset import EFFECTS



class PriorBoWModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.uniform_prior = args.uniform_prior
        self.entropy_regularize_prior_wt = args.entropy_regularize_prior_wt
        self.use_structured_prior = args.use_structured_prior
        self.use_structured_prior_binarypotential = args.use_structured_prior_binarypotential

        if not self.uniform_prior:
            self.roberta_embeddings = RobertaForSequenceClassification.from_pretrained('roberta-base').roberta.embeddings
            self.hidden_size = RobertaForSequenceClassification.from_pretrained('roberta-base').config.hidden_size
            self.history_tranformation = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            assert not self.entropy_regularize_prior_wt>0. # Doesn't make sense with uniform prior
            assert not self.use_structured_prior

        if self.use_structured_prior_binarypotential:
            assert self.use_structured_prior
        if self.use_structured_prior:
            self.emb_dim = args.effect_emb_dim #5
            self.effect_type_emb = nn.Embedding(len(EFFECTS), self.emb_dim)
            self.effect_type_to_feature = nn.Linear(self.emb_dim,1)
            if self.use_structured_prior_binarypotential:
                self.effecthistory_type_to_feature = nn.Linear(self.emb_dim+self.hidden_size, 1)
            self.num_feats = 2
            if self.use_structured_prior_binarypotential:
                self.num_feats += 1
            self.feature_combiner = nn.Parameter(torch.rand(self.num_feats).to(self.args.device))

    def get_prob_z_given_H(self, persona, history, effects=None):
        '''
        persona: B x P x T
        H: B x T

        We take the pooled output from Roberta; which uses same tokenization as GPT2
        '''

        persona = persona[..., 1:] # remove <s> token
        history = history[..., 1:] # remove <s> token

        if self.uniform_prior:
            num_persona = persona.shape[1]
            prob_z_given_H = torch.ones([persona.shape[0], persona.shape[1]]) / num_persona  # B x P
            return prob_z_given_H.to(self.args.device)

        else:
            history_encodings = self.roberta_embeddings(history).mean(dim=1)  # B x 764
            history_encodings = self.history_tranformation(history_encodings) # B x 764
            history_encodings = history_encodings.unsqueeze(1).repeat(1, persona.shape[1], 1)  # B x P x 764

            batch_size, num_personas, num_tokens = persona.shape
            persona = persona.reshape(-1, num_tokens)
            persona_encodings = self.roberta_embeddings(persona).mean(dim=1).reshape(batch_size, num_personas, -1)


            feats = -1.0 * torch.norm(history_encodings - persona_encodings, 2, dim=-1)
            # print("feats : ", feats.size())

            if self.use_structured_prior:
                embs = self.effect_type_emb(effects) # B,P,emsize
                effect_feature = self.effect_type_to_feature(embs) # B,P,1
                # print(torch.exp(effect_feature, dim=-1)) --> sort and analyze
                # print("feats : ", feats.size())
                # print("effect_feature : ", effect_feature.size())
                if self.use_structured_prior_binarypotential:
                    effecthistory_feature = self.effecthistory_type_to_feature(
                        torch.cat([embs,history_encodings],dim=2)) # B,P,1
                    feats = torch.cat([feats.unsqueeze(2), effect_feature,effecthistory_feature], dim=2)  # B,P,num_feats
                else:
                    feats = torch.cat([feats.unsqueeze(2), effect_feature], dim=2)  # B,P,num_feats
                feats = torch.sum( feats * self.feature_combiner.unsqueeze(0).unsqueeze(0), dim=2)

            prob_z_given_H = F.softmax(feats, dim=-1)
            # print("prob_z_given_H : ", prob_z_given_H.size())
            ret = prob_z_given_H # B x P

            return ret

    def sample(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: action, logprob of chosen action
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        action_idx = dist.sample()  # B
        return action_idx, dist.log_prob(action_idx)  #B;B

    def entropy(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: entropy
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        entropy = dist.entropy()  # B
        return entropy.mean() # 1
        # entropy_regularize_prior


class PriorRobertaModel(nn.Module):

    def __init__(self,
                 args):
        super().__init__()
        self.args = args
        self.uniform_prior = args.uniform_prior
        if not self.uniform_prior:
            self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', output_hidden_states=True)


    def get_prob_z_given_H(self, persona, history, effects=None):
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

            batch_size, num_personas, num_tokens = persona.shape
            persona = persona.reshape(-1, num_tokens)
            persona_encodings = self.roberta_model(persona)[1][-1][:, 0, :].reshape(batch_size, num_personas, -1)

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

    def entropy(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: entropy
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        entropy = dist.entropy()  # B
        return entropy.mean() # 1



class InferenceRobertaModel(nn.Module):

    def __init__(self,
                 args):
        super().__init__()
        self.args = args
        self.uniform_prior = args.uniform_prior
        if not self.uniform_prior:
            self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                                  output_hidden_states=True)
        self.use_history = False # TODO - add to args

    def get_prob_z_given_H_and_x(self, mc_token_ids, persona, history, effects=None):
        '''
        persona: B x P x T
        H: B x T
        We take the pooled output from Roberta; which uses same tokenization as GPT2
        TODO: Add <s> token at the beginning which will act as [CLS] token in BERT
        '''

        if self.uniform_prior:
            num_persona = persona.shape[1]
            prob_z_given_H_and_x = torch.ones([persona.shape[0], persona.shape[1]]) / num_persona  # B x P

            return prob_z_given_H_and_x.to(self.args.device)

        else:

            gt_response = mc_token_ids[:, :, 0]

            if self.use_history:
                history_encodings = self.roberta_model(history)[1][-1][:, 0, :]  # B x 764
                history_encodings = history_encodings.unsqueeze(1).repeat(1, persona.shape[1], 1)  # B x P x 764

            gt_response_encodings = self.roberta_model(gt_response)[1][-1][:, 0, :]  # B x 764

            batch_size, num_personas, num_tokens = persona.shape
            persona = persona.reshape(-1, num_tokens)
            persona_encodings = self.roberta_model(persona)[1][-1][:, 0, :].reshape(batch_size, num_personas, -1)
            if self.use_history:
                raise NotImplementedError

            norms = -1.0 * torch.norm(gt_response_encodings - persona_encodings, 2, dim=-1)
            prob_z_given_H_and_x = F.softmax(norms, dim=-1)

            return prob_z_given_H_and_x  # B x P

    def sample(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: action, logprob of chosen action
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        action_idx = dist.sample()  # B
        return action_idx, dist.log_prob(action_idx)  # B;B

    def entropy(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: entropy
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        entropy = dist.entropy()  # B
        return entropy.mean()  # 1

