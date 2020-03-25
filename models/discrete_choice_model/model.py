from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMarginalizedModel(nn.module):
    def __init__(self,
                 args,
                 encoder_class,
                 generator_class,
                 output_size,
                 hidden_dim,
                 max_len,
                 use_segments=False):
        super().__init__()

        self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', output_hidden_states=True)

        self.gpt2_model = generator_class.from_pretrained(args.model_checkpoint)
        self.criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.criterion_mc = torch.nn.CrossEntropyLoss(reduction='none')

    def get_prob_z_given_H(self, persona, history):
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

        norms = torch.norm(history_encodings-persona_encodings, 2, dim=-1)
        prob_z_given_H = F.softmax(norms, dim=-1)       

        return prob_z_given_H
    
    def forward(self, batch):
        '''
        persona: B x P x T
        input_ids: B x P x C x T
        mc_token_ids:
        lm_labels: B x P x C x T
        mc_labels: B
        token_type_ids: B x P x C x T 
        '''

        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, persona = batch

        history = []

        z_given_h = self.get_prob_z_given_H(persona, history) # B x P

        log_probs_lm = []
        log_probs_mc = []
        for i in range(input_ids.shape[1]):
            lm_logits, mc_logits, *_ = self.gpt2_model(
                input_ids[:, i, ...],
                token_type_ids=token_type_ids[:, i, ...],
                mc_token_ids=mc_token_ids[:, i, ...],
            )

            # LM
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            ll_lm = -1.0 * self.criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
            log_prob_x_given_z_h_lm = ll_lm + torch.log(z_given_h[:, i]) # B
            log_probs_lm.append(log_prob_x_given_z_h_lm)

            # MC
            ll_mc = -1.0 * self.criterion_mc(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            log_prob_x_given_z_h_mc = ll_mc + torch.log(z_given_h[:, i]) # B
            log_probs_mc.append(log_prob_x_given_z_h_mc)
        
        # LM
        log_probs_lm = torch.stack(log_prob_x_given_z_h_lm).T
        log_sum_exp_lm = torch.logsumexp(log_probs_lm) # logsumexp
        loss_lm = log_sum_exp_lm.sum()

        # MC
        log_probs_mc = torch.stack(log_prob_x_given_z_h_mc).T
        log_sum_exp_mc = torch.logsumexp(log_probs_mc) # logsumexp
        loss_mc = log_sum_exp_mc.sum()

        total_loss = loss_lm + loss_mc
        return total_loss