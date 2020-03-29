from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reinforce_model.prior_posterior_models import PriorModel
# from prior_posterior_models import PriorModel

TRAINING_TYPE_MARGINALIZE = 'marginalize'
TRAINING_TYPE_REINFORCE = 'reinforce'

class LatentMarginalizedModel(nn.Module):
    def __init__(self,
                 args,
                 generator_class):
        super().__init__()

        self.args = args
        self.prior_model = PriorModel(args)
        self.gpt2_model = generator_class.from_pretrained(args.model_checkpoint)
        self.criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.criterion_mc = torch.nn.CrossEntropyLoss(reduction='none')
        self.training_type = TRAINING_TYPE_MARGINALIZE #'marginalize' # or reinforce
        assert self.training_type in [TRAINING_TYPE_REINFORCE,TRAINING_TYPE_MARGINALIZE]

    def forward(
            self,
            input_ids,
            token_type_ids,
            persona=None,
            history=None,
            mc_token_ids=None,
            lm_labels=None,
            mc_labels=None,
            generate=False):
        '''
        persona: B x P x T
        input_ids: B x P x C x T
        mc_token_ids:
        lm_labels: B x P x C x T
        mc_labels: B
        token_type_ids: B x P x C x T
        '''

        if not generate:

            z_given_h = self.prior_model.get_prob_z_given_H(persona, history)  # B x P

            log_probs_lm = []
            log_probs_mc = []

            z_iterator = range(input_ids.shape[1])
            if self.training_type == TRAINING_TYPE_MARGINALIZE:
                z_iterator = range(input_ids.shape[1])
            elif self.training_type == TRAINING_TYPE_REINFORCE:
                action, logprob_action = self.prior_model.sample()
                z_iterator = [action] # in case of reinforce, do fwd for only one value of z
                z_given_h = z_given_h.detach() # do not uopdate prior through log likelihood since we are not marginalizing. we will instead udpate it through reinforce

            for i in z_iterator:
                lm_logits, mc_logits, *_ = self.gpt2_model(
                    input_ids[:, i, ...].contiguous(),
                    token_type_ids=token_type_ids[:, i, ...].contiguous(),
                    mc_token_ids=mc_token_ids[:, i, ...].contiguous(),
                )

                # LM
                lm_labels_persona = lm_labels[:, i, ...]
                mc_labels_persona = mc_labels[:, i, ...]
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels_persona[..., 1:].contiguous().view(-1)

                ll_lm = -1.0 * self.criterion_lm(lm_logits_flat_shifted, lm_labels_flat_shifted)  # B x C x T
                ll_lm = ll_lm.view(lm_labels.size(0), -1).mean(-1)  # B

                log_prob_x_given_z_h_lm = ll_lm + torch.log(z_given_h[:, i])  # B
                log_probs_lm.append(log_prob_x_given_z_h_lm)

                # MC
                ll_mc = -1.0 * self.criterion_mc(mc_logits.view(-1, mc_logits.size(-1)), mc_labels_persona.view(-1))
                ll_mc = ll_mc.view(mc_labels.size(0), -1).mean(-1)

                log_prob_x_given_z_h_mc = ll_mc + torch.log(z_given_h[:, i])  # B
                log_probs_mc.append(log_prob_x_given_z_h_mc)

            if self.training_type == TRAINING_TYPE_MARGINALIZE:
                # LM
                log_probs_lm = torch.stack(log_probs_lm).T  # B x P
                log_sum_exp_lm = torch.logsumexp(log_probs_lm, dim=1)  # logsumexp,  B
                loss_lm = -1.0 * log_sum_exp_lm.mean()

            elif self.training_type == TRAINING_TYPE_REINFORCE:
                # not when using reinforce, loss_lm is not log p(x) but log p(x\z=action) -- so be careful when compuing the perplexity
                # LM
                # log_probs_lm: P=1 values for B=batch_size. pick the first and only value
                log_probs_lm = log_probs_lm[0] #log_probs_lm:B
                log_sum_exp_lm = log_probs_lm # B
                loss_lm = -1.0 * log_sum_exp_lm.mean()
                # reward: we want to reward those actions which lead to higher
                rewards = log_sum_exp_lm.detach() # important to detach -> to not update the conditional model
                # todo - should do some sort of baseline computation for stable reinforce training
                loss_prior = - logprob_action * rewards # B
                loss_prior = loss_prior.mean() # B
                # sum the two losses. todo - use a weight on reinforce
                loss_lm = loss_lm + loss_prior

            # MC
            log_probs_mc = torch.stack(log_probs_mc).T
            log_sum_exp_mc = torch.logsumexp(log_probs_mc, dim=1)  # logsumexp
            loss_mc = -1.0 * log_sum_exp_mc.mean()

            return loss_lm, loss_mc


        if generate:
            lm_logits = self.gpt2_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )

            return lm_logits