from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from models.reinforce_model.utils import get_dataset, make_logdir
from models.reinforce_model.data import PADDED_INPUTS, ATTR_TO_SPECIAL_TOKEN
from models.reinforce_model.dataset import PersonaChatDataset, collate_dialog
from models.reinforce_model.train import add_special_tokens_
from models.reinforce_model.model import LatentMarginalizedModel

import torch
import math
import os

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
parser.add_argument("--model_checkpoint_dir", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--load_checkpoint_from", type=str, default="", help="Path, url or short name of the model")

parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
# parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
# parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
# parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for comet expansion")
parser.add_argument("--test_run_num", type=int, default=-1, help="Datapoints to run with in a test run")
# parser.add_argument("--exp_name", type=str, default="", required=True, help="Provide an experiment name")
parser.add_argument("--do_train", action='store_true', help="Do training")
parser.add_argument("--do_eval", action='store_true', help="Do Evaluation")
parser.add_argument("--no_persona", action='store_true', help="No Persona Evaluation")
parser.add_argument("--no_comet_persona", action='store_true', help="No Persona Evaluation")
parser.add_argument("--training_type", type=str, default="", help="Marginalize or Reinforce")
parser.add_argument("--prior_model", type=str, default="bow", help="Prior model selection")
args = parser.parse_args()


args.distributed = (args.local_rank != -1)

args.training_type = 'marginalize' # to make sure we are marginalizing 

training_args = torch.load(os.path.join(args.model_checkpoint_dir, 'model_training_args.bin'))
training_args.entropy_regularize_prior_wt = 0.0
print('Loaded training args.')

print("Prepare tokenizer, pretrained model and optimizer.")
tokenizer_class = GPT2Tokenizer # cant use Autotokenizer because checkpoint could be a Path
tokenizer = tokenizer_class.from_pretrained('gpt2')

orig_num_tokens = len(tokenizer.encoder)
print('Tokenizer length: {}'.format(orig_num_tokens))
num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
print('Tokenizer new length: {}'.format(len(tokenizer.encoder)))

model_class = GPT2DoubleHeadsModel
model = LatentMarginalizedModel(training_args, generator_class=model_class)
model.gpt2_model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

model_checkpoint_path = os.path.join(args.model_checkpoint_dir, args.load_checkpoint_from)
model_weights = torch.load(
        model_checkpoint_path, map_location=lambda storage, loc: storage
    )
model.load_state_dict(model_weights, strict=True)
print('Loaded model weights from {}'.format(model_checkpoint_path))

model.to(args.device)

# Add special tokens if they are not already added
# add_special_tokens_(model, tokenizer)

print("Prepare datasets")
start = datetime.now()

val_dataset = PersonaChatDataset(args, tokenizer, split='valid')

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=args.valid_batch_size,
    collate_fn=collate_dialog,
    pin_memory=True)

print('{} - Data loaded. Starting training'.format(datetime.now() - start))

num_correct = 0.0
num_examples = 0.0
ppls = []
losses = []

for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        # print(tokenizer.decode(input_ids[0, -1, :].tolist()))
        # if we dont send labels to model, it doesnt return losses
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, token_type_ids, lm_labels, mc_token_ids, mc_labels, persona, history, effects = batch
        
        (_), (_), (_), (marginal_lm_loss), (num_labels) = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            mc_token_ids=mc_token_ids,
            lm_labels=lm_labels,
            mc_labels=mc_labels,
            persona=persona,
            history=history
        )

        losses.append(marginal_lm_loss)

average_nll = sum(losses) / len(losses)
ppl = math.exp(average_nll)
print("Average Loss: {}".format(average_nll))
print("Average PPL: {}".format(ppl))

'''
/data2/bodhi/projects/persona-dialog/models/persona_weak_sup/runs/Mar03_01-49-47_deepyeti_gpt2weak_sup_og_persona

python3 -m models.reinforce_model.eval --dataset_path=/data3/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint_dir=/data3/bodhi/projects/persona-dialog/models/reinforce_model/runs/Apr09_15-26-28_deepx_gpt2prior_bow_high_lr_NEW/ --load_checkpoint_from=checkpoint_mymodel_130408.pth --lm_coef=2.0 --mc_coef=0.0 --max_history=2 --num_candidates=1 --personality_permutations=1 --valid_batch_size=1 --no_comet_persona --training_type=marginalize --test_run_num 30

w comet

python3 -m models.reinforce_model.eval --dataset_path=/data3/bodhi/data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json --model_checkpoint_dir=/data3/bodhi/projects/persona-dialog/models/reinforce_model/runs/Apr01_15-15-56_deepx_gpt2reinforce0.8_prior_bow_comet/ --load_checkpoint_from=checkpoint_mymodel_86940.pth --lm_coef=2.0 --mc_coef=0.0 --max_history=2 --num_candidates=1 --personality_permutations=1 --valid_batch_size=1 --training_type=marginalize --test_run_num 30

'''