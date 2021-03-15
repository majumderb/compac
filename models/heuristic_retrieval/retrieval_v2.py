from argparse import ArgumentParser
from tqdm import tqdm

from comet.matching_utils.weak_label_annotations import get_scores, get_recall_scores, process_text

import json

# load data

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='persona_comet_weak_label_preprocessed', help="Path or url of the dataset cache")
parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for comet expansion")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
parser.add_argument("--comet_persona", action='store_true')
parser.add_argument("--history", action='store_true')
parser.add_argument("--comet_history", action='store_true')
parser.add_argument("--use_recall_scores", action='store_true')


args = parser.parse_args()
print("args = ", args)
#0/0

personachat = json.load(open(args.dataset_path))
valid_data = personachat['valid']

correct = 0
top_3_corr = 0
total = 0
ranks = []
debug = False
good = 0
for d_i, dialog in tqdm(enumerate(valid_data), total=len(valid_data)):
    for u_i, utterance in enumerate(dialog['utterances']):
        grounding_doc = []
        # add personality
        grounding_doc += dialog["personality"]
        persona_len = len(dialog["personality"])

        if args.comet_persona:
            # add comet expansions for personality
            comet_annotations = dialog["coment_annotation"]
            sent_beams_persona = []
            for j_s, sent in enumerate(comet_annotations):
                # logging
                # if d_i == 0 and j_s == 0:
                #     print('For a sent: \n{}'.format(sent['comet']))
                for effect_name, effect in sent['comet'].items():
                    # if effect_name in EFFECTS:
                        # logging
                        # if d_i == 0 and j_s == 0:
                        #     print('Getting data for effect {}'.format(effect_name))
                        #     print('Getting {} beams'.format(len(effect['beams'][:args.num_beams])))
                        sent_beams_persona += effect['beams'][:args.num_beams]
            # if d_i == 0:
                # print('Got {} beams'.format(len(sent_beams_persona)))        
            grounding_doc += sent_beams_persona

        if args.history:
            # add history
            grounding_doc += utterance['history'][-(2*args.max_history+1):]

        if args.comet_history:
            # add comet expansions of history
            comet_history = dialog["history_comet_annotation"][:(2*u_i + 1)][-(2*args.max_history+1):]
            sent_beams_history = []
            for j_s, sent in enumerate(comet_history):
                # logging
                # if d_i == 0 and j_s == 0:
                #     print('For a sent: \n{}'.format(sent['comet']))
                for effect_name, effect in sent['comet'].items():
                    # if effect_name in EFFECTS:
                        # logging
                        # if d_i == 0 and j_s == 0:
                        #     print('Getting data for effect {}'.format(effect_name))
                        #     print('Getting {} beams'.format(len(effect['beams'][:args.num_beams])))
                        sent_beams_history += effect['beams'][:args.num_beams]
            # if d_i == 0:
                # print('Got {} beams'.format(len(sent_beams_history)))
            
            grounding_doc += sent_beams_history

        # grounding_doc = ' '.join(grounding_doc)
        grounding_doc = [process_text(s, typ='unigram') for s in grounding_doc]
        # print('GDs: {}'.format(len(grounding_doc)))

        grounding_doc_itr = grounding_doc

        candidate_scores = []
        for c_i, c in enumerate(utterance['candidates']):
            c = process_text(c, typ='unigram')
            candidate_doc_scores = []
            score_cur = 0.0
            for n, gd in enumerate(grounding_doc_itr):
                if args.use_recall_scores:
                    score = get_recall_scores(c, gd, 0)['score']
                else:
                    score = get_scores(c, gd, 0)['score']
                if n >= persona_len:
                    score = 0.8 * score
                score_cur += score
                #print(" ---> score_cur = ", score_cur)
                candidate_doc_scores.append((score, gd))

            candidate_doc_scores = sorted(candidate_doc_scores, key=lambda x: x[0], reverse=True)
            #candidate_scores.append((c_i, candidate_doc_scores[0] ))
            candidate_scores.append((c_i, (score_cur,"") ))

        gt_index = len(candidate_scores) - 1
        candidate_scores = sorted(candidate_scores, key=lambda x: x[1][0], reverse=True)
        print("candidate_scores = ", candidate_scores)

        correct += 1 if candidate_scores[0][0] == gt_index else 0
        curr_rank = [cs[0] for cs in candidate_scores].index(gt_index) + 1
        top_3_corr += 1 if curr_rank < 4 else 0
        # if curr_rank > 3:
            # print('Dialog: {}\nUtt: {}\nCandidate: {}\nGD: {}'.format(
                # d_i, u_i, c, grounding_doc))
            # debug = True
            # break
        ranks.append(curr_rank)
        total += 1
    if debug:
        break

print('Total {} utterances retrieved'.format(total))
print('Accuracy: {}'.format(correct / total))

print('Top-3 Accuracy: {}'.format(top_3_corr / total))

mrr = sum([1/r for r in ranks])/ len(ranks)
print('MRR: {}'.format(mrr))

print(good)


'''
python3 -m models.heuristic_retrieval.retrieval_harsh.py --dataset_path data/personachat/weak_label_comet_personachat/personachat_self_original_comet_scores_alignlabels.expanded_persona_history_preprocessed_validation.json --history --comet_persona
'''

        
'''
Accuracy: 0.34250243427458615
Top-3 Accuracy: 0.4870983446932814
MRR: 0.4429964462401029
0


Total 4108 utterances retrieved
Accuracy: 0.34250243427458615
Top-3 Accuracy: 0.4870983446932814
MRR: 0.4429964462401029
0


========== HISTORY 



========== persona 
Accuracy: 0.33325219084712754
Top-3 Accuracy: 0.48515092502434276
MRR: 0.43721311213398406
---Recall-scores:


========== persona + history
Accuracy: 0.3641674780915287
Top-3 Accuracy: 0.5805744888023369
MRR: 0.5072348348395014
---Recall-scores:
Accuracy: 0.28505355404089583
Top-3 Accuracy: 0.5464946445959105
MRR: 0.4544599847301807


========== persona + comet-persona + history
Accuracy: 0.26582278481012656
Top-3 Accuracy: 0.47151898734177217
MRR: 0.42386064154134523
---Recall-scores:

--------------------------------------  CUMULATIVE


========= persona + history : cumulative
Accuracy: 0.3758519961051607
Top-3 Accuracy: 0.5866601752677703
MRR: 0.5151546778980991



========== persona + comet-persona + history : cumulative
Accuracy: 0.12366114897760468
Top-3 Accuracy: 0.3473709834469328
MRR: 0.30569572821773855
'''
    
    