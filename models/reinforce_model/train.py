import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from functools import partial
from datetime import datetime

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from models.reinforce_model.model import LatentMarginalizedModel
from models.reinforce_model.utils import get_dataset, make_logdir
# from models.discrete_choice_model.data import get_data_loaders
from models.reinforce_model.dataset import PersonaChatDataset, collate_dialog, MAX_NUM_PERSONA, MAX_NUM_COMET_PERSONA
from models.reinforce_model.data import PADDED_INPUTS, ATTR_TO_SPECIAL_TOKEN

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

==
train w comet:
> python3 train.py --dataset_path=/data2/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint=gpt2 --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=1 --valid_batch_size=1 --test_run_num 5 --exp_name test --do_train --do_eval

'''

''' Notes:
Structured Prior:
--use_structured_prior -> to activate
--effect_emb_dim <intval>
'''


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
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
    parser.add_argument("--effect_emb_dim", type=int, default=6, help="Embedding type while computing effect feature")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print("Running process {}".format(args.local_rank))  # This is a logger.warning: it will be printed by all distributed processes
    print("Arguments: {}".format(pformat(args)))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    if args.do_eval and not args.do_train:
        print('Loading model from checkpoint {}'.format(args.model_checkpoint))
    # model = model_class.from_pretrained(args.model_checkpoint)
    # model.to(args.device)

    model = LatentMarginalizedModel(args, generator_class=model_class)
    model.to(args.device)

    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    print("Prepare datasets")
    start = datetime.now()

    train_dataset = PersonaChatDataset(args, tokenizer, split='train')
    if args.do_eval:
        val_dataset = PersonaChatDataset(args, tokenizer, split='valid')

    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    if args.no_comet_persona:
        max_num_persona = MAX_NUM_PERSONA
    else:
        max_num_persona = MAX_NUM_COMET_PERSONA
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=partial(collate_dialog, max_num_persona=max_num_persona),
        pin_memory=True)
    
    if args.do_eval:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.valid_batch_size,
            collate_fn=partial(collate_dialog, max_num_persona=max_num_persona),
            pin_memory=True)

    print('{} - Data loaded. Starting training'.format(datetime.now() - start))
    # train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        
        model.train()

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, token_type_ids, lm_labels, mc_token_ids, mc_labels, persona, history, effects = batch
        
        (lm_loss), (mc_loss), (loss_prior), (conditional_lm_loss), (num_labels) = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=mc_token_ids,
            lm_labels=lm_labels,
            mc_labels=mc_labels,
            persona=persona,
            history=history,
            effects=effects
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

        return loss.item(), lm_loss.item(), mc_loss.item(), loss_prior.item(), conditional_lm_loss.item()
    
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
    
        model.eval()
    
        with torch.no_grad():
    
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
    
            # print(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
    
            lm_logits, mc_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
    
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
    
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    
    evaluator = Engine(inference)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    # if args.distributed:
    #     trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
    #     evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lm_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "mc_loss")
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, "prior_loss")
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, "cond_lm_loss")

    metrics = {
        "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
        "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}

    metrics.update(
        {"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
        "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})

    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    def print_model_save(engine):
        print("Training complete. Saving Model.")
    
    def print_validation(engine):
        print("Model saved. Starting validation.")

    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss", "lm_loss", "mc_loss", "prior_loss", "cond_lm_loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint, args.exp_name)
        log_dir = os.path.join(args.log_dir, log_dir)

        print("Logging at log dir: {}".format(log_dir))

        # tb stuff
        # tb_logger = TensorboardLogger(log_dir)
        # tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        # tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        # tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        # save model checkpoints
        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=None)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, print_model_save)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        # getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

        # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, print_validation)
        if args.do_eval:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
            if args.n_epochs < 1:
                trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
            if args.eval_before_start:
                trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Run the training
    if args.do_train:
        trainer.run(train_loader, max_epochs=args.n_epochs)
    if args.do_eval and not args.do_train:
        print('Running only Evaluation. No Training.')
        evaluator.run(val_loader)


    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0 and args.do_train:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        # tb_logger.close()

if __name__ == "__main__":
    train()