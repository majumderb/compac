import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import os
import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from models.discrete_choice_model.train import add_special_tokens_
from models.discrete_choice_model.data import SPECIAL_TOKENS, build_input_from_segments, ATTR_TO_SPECIAL_TOKEN
from models.discrete_choice_model.dataset import ROBERTA_START
from models.discrete_choice_model.utils import get_dataset, download_pretrained_model
from models.discrete_choice_model.model import LatentMarginalizedModel

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    # 1 x P x T
    personality = torch.tensor([[ROBERTA_START] + p  for p in personality], device=args.device).unsqueeze(0) 
    # 1 x T
    history_flat = torch.tensor([ROBERTA_START] + list(chain(*history)), device=args.device).unsqueeze(0)

    prior_z = model.get_prob_z_given_H(personality, history_flat) # B x P
    z = torch.argmax(prior_z, dim=1).item()

    selected_personality = personality[z]

    for i in range(args.max_length):
        instance = build_input_from_segments(selected_personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids, generate=True)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

'''
python3 interact.py --dataset_path=/data2/bodhi/data/personachat/comet_persona_outputs_v1/personachat_self_original_comet.json --model gpt2 --model_checkpoint runs/Feb22_13-26-13_deepyeti_gpt2concat_comet/
'''

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='persona_comet', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint_dir", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--load_checkpoint_from", type=str, default="", help="Path, url or short name of the model")

    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--comet_greedy", action='store_true', help="Use top-most comet expansion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    logger.info("Get finetuned model and tokenizer")
    training_args = torch.load(os.path.join(args.model_checkpoint_dir, 'model_training_args.bin'))
    print('Loaded training args.')
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint_dir)
    print('Tokenizer length: {}'.format(len(tokenizer.encoder)))
    
    model = LatentMarginalizedModel(training_args, generator_class=model_class)
    add_special_tokens_(model, tokenizer)

    # Load model weights
    model_checkpoint_path = os.path.join(args.model_checkpoint_dir, args.load_checkpoint_from)
    model_weights = torch.load(
        model_checkpoint_path, map_location=lambda storage, loc: storage
    )
    # corrected_model_weights = {}
    # for k, v in model_weights.items():
    #     new_k = k.replace('gpt2_model.', '').replace('', '')
    #     corrected_model_weights[k.replace('gpt2_model.', '')] = v

    model.load_state_dict(model_weights, strict=False)
    print('Loaded model weights from {}'.format(model_checkpoint_path))

    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    # logger.info("Sample a personality")
    # dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    # # personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    # dialogs = [dialog for dataset in dataset.values() for dialog in dataset]
    # dialog =  random.choice(dialogs)
    # # personality = random.choice(personalities)
    # personality = dialog['personality']
    # comet_annotations = dialog["coment_annotation"]
    # for sent in comet_annotations:
    #     sent_beams = []
    #     for effect in sent['comet'].items():
    #         # not sure is ' .' should be added or '.'
    #         # tokenizer realize different tokens for each of the above options
    #         # beams = [x+' .' for x in effect[1]['beams']]
    #         if args.comet_greedy:
    #             sent_beams += [effect[1]['beams'][0]]
    #         else:
    #             sent_beams += effect[1]['beams']
    # personality += sent_beams
    # print(personality)
    # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    # history = []
    # while True:
    #     raw_text = input(">>> ")
    #     while not raw_text:
    #         print('Prompt should not be empty!')
    #         raw_text = input(">>> ")
    #     history.append(tokenizer.encode(raw_text))
    #     with torch.no_grad():
    #         out_ids = sample_sequence(personality, history, tokenizer, model, args)
    #     history.append(out_ids)
    #     history = history[-(2*args.max_history+1):]
    #     out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    #     print(out_text)


if __name__ == "__main__":
    run()