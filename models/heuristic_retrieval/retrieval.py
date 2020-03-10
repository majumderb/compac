from argparse import ArgumentParser
from tqdm import tqdm

from comet.matching_utils.weak_label_annotations import get_scores, get_recall_scores, process_text

import json

# load data

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for comet expansion")

args = parser.parse_args()

personachat = json.load(open(args.dataset_path))
valid_data = personachat['valid']

correct = 0
total = 0
ranks = []
for d_i, dialog in tqdm(enumerate(valid_data), total=len(valid_data)):
    for u_i, utterance in enumerate(dialog['utterances']):
        grounding_doc = []
        # add personality
        grounding_doc += dialog["personality"]

        # add comet expansions for personality
        comet_annotations = dialog["coment_annotation"]
        sent_beams_persona = []
        for j_s, sent in enumerate(comet_annotations):
            # logging
            if d_i == 0 and j_s == 0:
                print('For a sent: \n{}'.format(sent['comet']))
            for effect_name, effect in sent['comet'].items():
                # if effect_name in EFFECTS:
                    # logging
                    if d_i == 0 and j_s == 0:
                        print('Getting data for effect {}'.format(effect_name))
                        print('Getting {} beams'.format(len(effect['beams'][:args.num_beams])))
                    sent_beams_persona += effect['beams'][:args.num_beams]
        if d_i == 0:
            print('Got {} beams'.format(len(sent_beams_persona)))        
        grounding_doc += sent_beams_persona

        # add history
        grounding_doc += utterance['history']

        # add comet expansions of history
        comet_history = dialog["history_comet_annotation"][:(2*u_i + 1)]
        sent_beams_history = []
        for j_s, sent in enumerate(comet_history):
            # logging
            if d_i == 0 and j_s == 0:
                print('For a sent: \n{}'.format(sent['comet']))
            for effect_name, effect in sent['comet'].items():
                # if effect_name in EFFECTS:
                    # logging
                    if d_i == 0 and j_s == 0:
                        print('Getting data for effect {}'.format(effect_name))
                        print('Getting {} beams'.format(len(effect['beams'][:args.num_beams])))
                    sent_beams_history += effect['beams'][:args.num_beams]
        if d_i == 0:
            print('Got {} beams'.format(len(sent_beams_history)))
        
        grounding_doc += sent_beams_history

        grounding_doc = ' '.join(grounding_doc)
        grounding_doc = process_text(grounding_doc)

        candidate_scores = []
        for c_i, c in enumerate(utterance['candidates']):
            c = process_text(c)
            score = get_recall_scores(c, grounding_doc, 0)
            candidate_scores.append((c_i, score['score']))
        
        gt_index = len(candidate_scores) - 1
        candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        correct += 1 if candidate_scores[0][0] == gt_index else 0
        ranks.append([cs[0] for cs in candidate_scores].index(gt_index) + 1)
        total += 1

print('Total {} utterances retrieved'.format(total))
print('Accuracy: {}'.format(correct / total))

mrr = sum([1/r for r in ranks])/ len(ranks)
print('MRR: {}'.format(mrr))


        




        