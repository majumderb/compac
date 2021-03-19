import os
import math
import logging
import random
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from functools import partial
from datetime import datetime

import numpy as np
import torch
from apex import amp
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from ignite.engine import Engine, Events
from ignite.exceptions import NotComputableError
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, Metric, MetricsLambda, RunningAverage
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from models.reinforce_model.model_with_inferencenw import LatentVariableInferenceModel
from models.reinforce_model.utils import get_dataset, make_logdir
# from models.discrete_choice_model.data import get_data_loaders
from models.reinforce_model.dataset import PersonaChatDataset, MAX_NUM_PERSONA, MAX_NUM_COMET_PERSONA
from models.reinforce_model.data import PADDED_INPUTS, ATTR_TO_SPECIAL_TOKEN


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class Perplexity(Metric):
    def __init__(self, output_transform=lambda x: x, validate_args=True):
        self.validate_args = validate_args
        super(Perplexity, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._num_distributions = torch.tensor(0, dtype=torch.int)
        self._perplexities_sum = torch.tensor(0.0, dtype=torch.float)

    @reinit__is_reduced
    def update(self, output):
        y_pred, _ = output
        if y_pred.ndimension() != 2:
            raise ValueError(f"Predictions (logits or probabilities has to have 2 dimensins. Got y_pred.shape={y_pred.shape}")
        d = Categorical(None, y_pred, validate_args=self.validate_args)
        ppl = d.perplexity()
        self._perplexities_sum += ppl.sum()
        self._num_distributions += ppl.numel()

    @sync_all_reduce("_num_distributions", "_perplexities_sum") 
    def compute(self):
        if self._num_distributions.eq(0):
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        ppl = self._perplexities_sum / self._num_distributions
        return ppl


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    # if num_added_tokens > 0:
    model.gpt2_model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


'''
deepy

train:
TEST run
> python3 -m models.reinforce_model.train --dataset_path=/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint=gpt2 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=2 --n_epochs=1 --num_candidates=1 --personality_permutations=1 --train_batch_size=1 --valid_batch_size=1 --no_comet_persona --do_train --training_type=reinforce --use_baseline --moving_avg_ratio=0.99 --reinforce_loss_coef=0.99  --test_run_num 1 --log_dir models/reinforce_model/ --exp_name reinforce_TEST

Final run
python3 -m models.reinforce_model.train --dataset_path=/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint=gpt2 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=2 --n_epochs=1 --num_candidates=1 --personality_permutations=1 --train_batch_size=1 --valid_batch_size=1 --no_comet_persona --do_train --training_type=reinforce --use_baseline --moving_avg_ratio=0.99 --reinforce_loss_coef=0.99 --log_dir models/reinforce_model/ --exp_name reinforce_TEST

deep x 
python3 -m models.reinforce_model.train --dataset_path=/data3/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint=gpt2 --gradient_accumulation_steps=4 --lm_coef=2.0 --mc_coef=0.0 --max_history=2 --n_epochs=10 --num_candidates=1 --personality_permutations=1 --train_batch_size=2 --valid_batch_size=2 --do_train --training_type=reinforce --use_baseline --moving_avg_ratio=0.99 --reinforce_loss_coef=0.8 --lr=1e-4 --log_dir models/reinforce_model/ --exp_name EXP_NAME

==
train w comet:
> python3 train.py --dataset_path=/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint=gpt2 --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=1 --valid_batch_size=1 --test_run_num 5 --exp_name test --do_train --do_eval

'''

''' Notes:
Structured Prior:
--use_structured_prior -> to activate
--effect_emb_dim <intval>
'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, help="Path to `LatentVariableInferenceModel` weights.")
    parser.add_argument(
        "--generation_model", 
        default="openai-gpt", 
        help="The name of generator model. Available options are `gpt2` and `openai-gpt`. Default is `openai-gpt`.", 
        choices=["gpt2", "openai-gpt"],
    )
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for comet expansion")
    parser.add_argument("--test_run_num", type=int, default=-1, help="Datapoints to run with in a test run")
    parser.add_argument("--exp_name", type=str, default="", required=True, help="Provide an experiment name")
    parser.add_argument("--do_train", action='store_true', help="Do training")
    parser.add_argument("--do_eval", action='store_true', help="Do Evaluation")
    parser.add_argument("--no_persona", action='store_true', help="No Persona Evaluation")
    parser.add_argument("--no_comet_persona", action='store_true', help="No Persona Evaluation")
    parser.add_argument("--uniform_prior", action='store_true', help="Uniform prior")
    parser.add_argument("--entropy_regularize_prior_wt", type=float , default=0.0, help="entropy regularize prior")
    parser.add_argument("--training_type", type=str, default="", help="Marginalize or Reinforce")
    parser.add_argument("--use_baseline", action='store_true', help="Use baseline")
    parser.add_argument("--moving_avg_ratio", type=float, default=0.99, help="Moving avg ratio for running mean baseline")
    parser.add_argument("--reinforce_loss_coef", type=float, default=0.99, help="Loss coef for reinforce")
    parser.add_argument("--prior_model", type=str, default="bow", help="Prior model selection")
    parser.add_argument("--log_dir", type=str, default="", required=True, help="Provide a log dir")
    parser.add_argument("--use_structured_prior", action='store_true', default=False, help="Use effect type as feature")
    parser.add_argument("--use_structured_prior_binarypotential", action='store_true', default=False, help="")
    parser.add_argument("--effect_emb_dim", type=int, default=6, help="Embedding type while computing effect feature")
    args = parser.parse_args()
    if not args.do_train and args.do_eval:
        raise ValueError("You have to specify at least one of options `--do_train`, `--do_eval`")
    return args


def create_evaluator(args, model):
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = {name: input_tensor.to(args.device) for name, input_tensor in batch.items()}
            lm_logits, mc_logits, *_ = model(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"],
                mc_token_ids=batch["mc_token_ids"],
                lm_labels=batch["lm_labels"],
                mc_labels=batch["mc_labels"],
                persona=batch["persona"],
                history=batch["history"],
                effects=batch["effects"],
            )
            # print("(inference)mc_logits.shape, mc_labels.shape:", mc_logits.shape, batch["mc_labels"].shape)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = batch["lm_labels"][:, 0, :, 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, batch["mc_labels"])
    
    evaluator = Engine(inference)
    metrics = {
        "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
        # the accuracy is a filler since multiple-choice is not used.
        "accuracy": Accuracy(
            output_transform=lambda x: (torch.argmax(x[0][1].view((-1,)), dim=0, keepdim=True), x[1][1][:, 0])),
        "ppl": Perplexity(output_transform=lambda x: (x[0][0], None)),
    }

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(evaluator)
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))
    return evaluator


def create_trainer_and_checkpoint_handler(args, model, optimizer, train_loader, val_loader, evaluator, log_dir):
    def update(engine, batch):        
        model.train()
        batch = {name: input_tensor.to(args.device) for name, input_tensor in batch.items()}
        _, _, lm_loss, mc_loss, loss_prior, conditional_lm_loss, num_labels, track_rewards, kl_loss, elbo_loss_tracking = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            mc_token_ids=batch["mc_token_ids"],
            lm_labels=batch["lm_labels"],
            mc_labels=batch["mc_labels"],
            persona=batch["persona"],
            history=batch["history"],
            effects=batch["effects"],
        )
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), lm_loss.item(), mc_loss.item(), loss_prior.item(), conditional_lm_loss.item(), track_rewards.item(), kl_loss.item(), elbo_loss_tracking.item()
    
    trainer = Engine(update)
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lm_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "mc_loss")
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, "prior_loss")
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, "cond_lm_loss")
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, "rewards")
    RunningAverage(output_transform=lambda x: x[6]).attach(trainer, "kl_loss")
    RunningAverage(output_transform=lambda x: x[7]).attach(trainer, "elbo_loss")
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss", "lm_loss", "mc_loss", "prior_loss", "cond_lm_loss", "rewards", "kl_loss", "elbo_loss"])
        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=None)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda: print("Training complete. Saving Model."))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda: print("Model saved. Starting validation."))
    else:
        checkpoint_handler = None
    if args.do_eval:
        assert evaluator is not None
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
        if args.n_epochs < 1:
            trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
        if args.eval_before_start:
            trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))
    return trainer, checkpoint_handler


def create_model_and_optimizer(args, tokenizer):
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.generation_model else OpenAIGPTDoubleHeadsModel

    model = LatentVariableInferenceModel(args, generator_class=model_class)
    print('Num parameters: {}'.format(count_parameters(model)))
    model.to(args.device)
    if args.model_checkpoint is not None:
        print('Loading model from checkpoint {}'.format(args.model_checkpoint))
        model_weights = torch.load(args.model_checkpoint, map_location=args.device)
        model.load_state_dict(model_weights)
    add_special_tokens_(model, tokenizer)
    if args.do_train:
        optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    else:
        optimizer = None
    if args.fp16 and args.do_train:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.local_rank != -1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    return model, optimizer   


def create_tokenizer(args):
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.generation_model else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.generation_model)
    return tokenizer


def init_distributed(args):
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


def create_val_dataloader(args, tokenizer, max_num_persona):
    val_dataset = PersonaChatDataset(args, tokenizer, split='valid')
    if args.local_rank == -1:
        val_sampler = SequentialSampler(val_dataset)
    else:
        val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.valid_batch_size,
        collate_fn=partial(val_dataset.collate_dialog, max_num_persona=max_num_persona),
        pin_memory=True,
        sampler=val_sampler,
    )
    return val_loader


def create_train_dataloader(args, tokenizer, max_num_persona):
    train_dataset = PersonaChatDataset(args, tokenizer, split='train')
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=partial(train_dataset.collate_dialog, max_num_persona=max_num_persona),
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return train_loader


def train():
    args = get_args()
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print("Running process {}".format(args.local_rank))
    print("Arguments: {}".format(pformat(args)))

    if args.local_rank != -1:
        init_distributed(args)
    print("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer = create_tokenizer(args)
    model, optimizer = create_model_and_optimizer(args, tokenizer)

    print("Prepare datasets")
    start = datetime.now()
    max_num_persona = MAX_NUM_PERSONA if args.no_comet_persona else MAX_NUM_COMET_PERSONA

    log_dir = make_logdir(args.generation_model, args.exp_name)
    log_dir = os.path.join(args.log_dir, log_dir)
    if args.local_rank in [-1, 0]:
        os.makedirs(log_dir, exist_ok=True)
        print("Logging at log dir: {}".format(log_dir))
        torch.save(args, log_dir + '/model_training_args.bin')
        tokenizer.save_pretrained(log_dir)

    if args.do_eval:
        val_loader = create_val_dataloader(args, tokenizer, max_num_persona)
        evaluator = create_evaluator(args, model)
    else:
        val_loader, evaluator = None, None
    if args.do_train:
        train_loader = create_train_dataloader(args, tokenizer, max_num_persona)
        trainer, checkpoint_handler = create_trainer_and_checkpoint_handler(args, model, optimizer, train_loader, val_loader, evaluator, log_dir)
    else:
        train_loader, trainer = None, None

    print('{} - Data loaded. Starting training'.format(datetime.now() - start))


    if args.do_train:
        trainer.run(train_loader, max_epochs=args.n_epochs)
    if args.do_eval and not args.do_train:
        print('Running only Evaluation. No Training.')
        evaluator.run(val_loader)
    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0 and args.do_train:
        assert checkpoint_handler is not None
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        # tb_logger.close()

if __name__ == "__main__":
    np.random.seed(42)
    #torch.use_deterministic_algorithms()
    train()
