import json
from collections import defaultdict 

with open('personachat_self_original_comet_scores_alignlabels.expanded_persona_preprocessed.json', "r+", encoding="utf-8") as f: 
    dataset = json.loads(f.read())['train'] 

effects = defaultdict(float)

for i in range(len(dataset)): 
    for sen in dataset[i]['weak_labels_comet']: 
        if len(sen['label_persona']) > 0: 
            for l in sen['label_persona']: 
                effects[l[0]['comet_key']] += 1

factor=1.0/sum(effects.values()) 
    for k in effects: 
        effects[k] = effects[k]*factor

'''
# Sorted
oReact: 0.04397302955065479
xReact: 0.05377050776495789
xEffect: 0.06250912808529283
xAttr: 0.06538143225743635
oWant: 0.11651696606786427
xWant: 0.1188902682439998
xIntent: 0.16486539116888174
oEffect: 0.17482108952826056
xNeed: 0.19927218733265176
'''


'''
Weak label

'oReact', 'xReact', 'xEffect', 'xAttr', 'oWant', 'xWant', 'xIntent', 'xNeed', 'oEffect'

GlobalEff

Ep 0: 'xAttr', 'oReact', oEffect, 'xNeed', 'oWant', 'xEffect', 'xReact', 'xWant', 'xIntent'
Ep 3: 'oReact', 'oEffect', 'oWant', 'xEffect', 'xReact', 'xWant', 'xAttr', 'xIntent', 'xNeed'
Ep 7: 'oReact', 'xEffect', 'xReact', 'oWant', 'xWant', 'xAttr', 'xIntent', 'xNeed', 'oEffect'

HistEff

Ep 0: 'oReact', 'xEffect', 'oEffect', 'oWant', 'xReact', 'xWant', 'xAttr', 'xIntent', 'xNeed'
Ep 3: 'oReact', 'oWant', 'xEffect', 'xReact', 'oEffect', 'xWant', 'xAttr', 'xIntent', 'xNeed'
Ep 7: 'oReact', 'xEffect', 'xAttr', 'xReact', 'xWant', 'oWant', 'xIntent', 'oEffect', 'xNeed'
'''