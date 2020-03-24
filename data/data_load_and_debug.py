import json
data = json.load(open('personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json'))
print(data.keys())
print(data['train'][0].keys())
