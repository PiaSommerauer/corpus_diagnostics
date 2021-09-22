from itertools import permutations
import os
import numpy as np
import pandas as pd

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