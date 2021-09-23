from itertools import permutations
from collections import defaultdict

import os
import numpy as np
import pandas as pd
import json


def load_relation_pairs(combination, order=True):
    
    all_pairs = set()
    
    path_dir = '../data/relations'
    if order == True:
        name = '-'.join(combination)
        paths = [f'{path_dir}/{name}.txt']
    else:
        perms = list(permutations(combination, len(combination)))
        paths = []
        for perm in perms:
            name = '-'.join(perm)
            paths.append(f'{path_dir}/{name}.txt')
    
    for path in paths:
        if os.path.isfile(path):
            with open(path) as infile:
                lines = infile.read().strip().split('\n')
            pairs = [(l.split(',')[0], l.split(',')[1]) for l in lines]
            all_pairs.update(pairs)
    return all_pairs


def str_to_tuple(pair_str):
    
    pair_str = pair_str.strip('(').strip(')').replace("'", '')
    pair_list = pair_str.split(', ')
    pair_t = tuple(pair_list)
    return pair_t


def load_scores(analysis_name, model_name):

    path_dir = f'../analysis/{model_name}/pairs/'
    path_file = f'{path_dir}/{analysis_name}.csv'
    df = pd.read_csv(path_file)#.fillna(0.0)

    pair_score_dict = dict()
    for i, row in df.iterrows():
        pair = row['pair']
        pair = str_to_tuple(pair)
        score = row['prop-specific'] 
        pair_score_dict[pair] = score
    return pair_score_dict


def relation_overview(pair_score_dict):

    relations =  ['pos', 'neg', 'all', 'some', 'few',
                         'evidence', 'no_evidence_pos', 'no_evidence_neg',
                         'implied_category', 
                         'typical_of_concept', 'typical_of_property', 
                         'affording_activity', 'afforded_usual', 'afforded_unusual',
                         'variability_limited', 'variability_open',
                         'variability_limited_scalar', 'variability_open_scalar',
                         'rare', 'unusual', 'impossible', 'creative']

    table = []
    for rel in relations:
        pairs = load_relation_pairs([rel], order=False)
        scores = []
        for pair in pairs:
            if pair in pair_score_dict.keys():
                scores.append(pair_score_dict[pair])
        if len(scores) > 0:
            med = np.nanmedian(scores)
            d = dict()
            d['relation'] = rel
            d['prop-specific'] = med
            d['n_pairs'] = len(scores)
            table.append(d)

    df = pd.DataFrame(table).set_index('relation')
    return df



def relations_to_file(properties):
    path_dir = '../data/relations'
    os.makedirs(path_dir, exist_ok=True)
    relation_pair_dict = defaultdict(set)
    for prop in properties:
        prop_dict = utils.load_prop_data(prop)
        rels_evidence = {'typical_of_property', 'variability_limited',
                         'afforded_usual', 'affording_activity', 
                         'variability_limited_scalar', 'variability_open_scalar'}
        rels_no_evidence = {'typical_of_concept', 'afforded_unusual', 'implied_category', 'variability_open'}

        for c, d in prop_dict.items():
            # sort by lables
            ml_label = d['ml_label']
            if ml_label in {'all', 'some', 'all-some', 'few-some'}:
                l = 'pos'
            elif ml_label in {'few'}:
                l = 'neg' 
            relation_pair_dict[(l,)].add((prop, c))

            # sort by subset of instances
            if ml_label in {'all', 'all-some'}:
                l_qu = 'all'
            elif ml_label in {'some', 'few-some'}:
                l_qu = 'some'
            elif ml_label in {'few'}:
                l_qu = 'few'
            relation_pair_dict[(l_qu,)].add((prop, c))

            # sort relations:
            rel_dict = d['relations']
            prop_rel_dict = defaultdict(list)
            for rel, p in rel_dict.items():
                prop_rel_dict[p].append(rel)
            pos_props = sorted(list(prop_rel_dict.keys()), reverse=True)
            relations = []
            for p in pos_props:
                if p > 0.5:
                    relations.extend(prop_rel_dict[p])
            relations = tuple(relations)
            relation_pair_dict[relations].add((prop, c))

            # sort relations according to hypotheses
            relations = [rel for rel, p in rel_dict.items() if p > 0.5]
            rel_ev = [rel for rel in relations if rel in rels_evidence]
            rel_no_ev  = [rel for rel in relations if rel in rels_no_evidence]

            if len(rel_ev) > 0:
                hyp = 'evidence'
            elif len(rel_no_ev) > 0:
                hyp = 'no_evidence_pos'
            else:
                hyp = 'no_evidence_neg'
            relation_pair_dict[(hyp, )].add((prop, c))

        # to file
        for rel, pairs in relation_pair_dict.items():
            
            name = '-'.join(rel)
            path = f'{path_dir}/{name}.txt'
            with open(path,  'w') as outfile:
                for prop, c in pairs:
                    outfile.write(f'{prop},{c}\n')
                    
                    
if __name__ == '__main__':
                    
    properties = utils.get_properties()  
    # exclude female from this analysis
    properties = [prop for prop in properties  if prop != 'female']
    relations_to_file(properties)