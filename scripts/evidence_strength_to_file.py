import analyze_annotations
import analyze_evidence

import os
import sys

def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated_semantic_info/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties



# extract mean evidence strength for context words that give prop-specific evidence
model_name = sys.argv[1]
properties = get_properties()


top_cutoff = 3
concept_cutoff = 3

ets = ['p', 'n', 'l', 'i', 'r', 'b']


for prop in properties:
    print(prop)
    table = analyze_evidence.get_top_ev_categories(prop, model_name, top_cutoff, concept_cutoff)
    for et_target in ets:
        contexts = set()
        for (cat, et), d in table.items():
            if et_target == et:
                contexts.update(set(d['contexts'].split(' ')))
        if len(contexts) > 0:
            analyze_annotations.evidence_strength_to_file(prop, et_target, contexts, model_name, top_cutoff, concept_cutoff)