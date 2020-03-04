from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from argparse import ArgumentParser
from tqdm import tqdm

from utils import get_dataset, make_logdir
from data import get_data_loaders
from data import PADDED_INPUTS, ATTR_TO_SPECIAL_TOKEN
from train import add_special_tokens_

import torch

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
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
args = parser.parse_args()


args.distributed = (args.local_rank != -1)

print("Prepare tokenizer, pretrained model and optimizer.")
tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
print('Loading model from checkpoint {}'.format(args.model_checkpoint))
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)

# Add special tokens if they are not already added
add_special_tokens_(model, tokenizer)

print("Prepare datasets")
train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

num_correct = 0.0
num_examples = 0.0
for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
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

        # collect multiple choice (mc) logits for accuracy
        indices = torch.argmax(mc_logits, dim=1)
        correct = torch.eq(indices, mc_labels).view(-1)
    
    num_correct += torch.sum(correct).item()
    num_examples += correct.shape[0]

print("Accuracy: {}".format(num_correct / num_examples))

# (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)


"""
/data2/bodhi/projects/persona-dialog/models/persona_weak_sup/runs/Mar03_01-49-47_deepyeti_gpt2weak_sup_og_persona
"""