from collections import defaultdict, Counter
import os
import csv
import numpy as np


def get_annotation_status(model_name, top_cutoff, concept_cutoff):
    dir_path = f'../analysis/{model_name}'
    dir_annotations = f'{dir_path}/annotation-tfidf-top_{top_cutoff}_{concept_cutoff}-raw-10000-categories'
    annotation_dict = defaultdict(set)
    line_dict = dict()

    for f in os.listdir(dir_annotations):
        if  not f.endswith('.csv') and not f.endswith('.ipynb_checkpoints') and not f.endswith('.DS_Store'):
            prop = f.split('/')[-1]
            full_path = f'{dir_annotations}/{f}'
            
            #print(full_path)
            # get categories:
            files = os.listdir(full_path)
            # get number of words
            if model_name == 'wiki_updated':
                path_file = f'{full_path}/annotation-transferred-updated.csv'
            elif model_name == 'giga_full_updated':
                path_file = f'{full_path}/annotation-updated.csv'
            with open(path_file) as infile:
                lines = infile.read().strip().split('\n')
                not_annotated = [l for l in lines if l.strip().split(',')[1] == 'NA']
            line_dict[prop] = (len(lines), len(not_annotated), len(lines)-len(not_annotated))
            if 'annotation-updated-done.csv' in files:
                annotation_dict['complete'].add(prop)
            else:
                annotation_dict['incomplete'].add(prop)
                
    return annotation_dict, line_dict

def show_annotation_status(model_name, top_cutoff, concept_cutoff):
    annotation_dict, line_dict = get_annotation_status(model_name, 
                                        top_cutoff, concept_cutoff)
    # same category not annotated:
    print('completed:\n')
    for prop in sorted(list(annotation_dict['complete'])):
        # cats open:
        print(prop, line_dict[prop])
    print()
    print('Incomplete:\n')
    for prop in sorted(annotation_dict['incomplete']):
        if prop not in annotation_dict['complete']:
            print(prop, line_dict[prop])
    return annotation_dict
            
            
def get_evidence_dict(model_name, prop, top_cutoff, concept_cutoff):
    
    annotation_name = f'annotation-tfidf-top_{top_cutoff}_{concept_cutoff}-raw-10000-categories'
    path_dir_annotation = f'../analysis/{model_name}/{annotation_name}/{prop}'
    f_annotation = f'{path_dir_annotation}/annotation-updated-done.csv'
    
    ev_dict = dict()
    
    with open(f_annotation) as infile:
        data = list(csv.DictReader(infile))
    for d in data:
        et = d['evidence_type']
        ev = d['context']
        ev_dict[ev] = et
    return ev_dict
        
            

def get_evidence_density(model_name, prop, top_cutoff, concept_cutoff):
    
    # current file:
    annotation_name = f'annotation-tfidf-top_{top_cutoff}_{concept_cutoff}-raw-10000-categories'
    path_dir_annotation = f'../analysis/{model_name}/{annotation_name}/{prop}'
    f_annotation = f'{path_dir_annotation}/annotation-updated-done.csv'
    
    ev_dict = get_evidence_dict(model_name, prop, top_cutoff, concept_cutoff)
    
    ev_cnts = Counter()
    
    for e, et in ev_dict.items():
        ev_cnts[et] += 1
        if et != 'u':
            ev_cnts['all'] += 1
        if et in ['p', 'l', 'n']:
            ev_cnts['prop_specific'] += 1
        elif et in ['i', 'r', 'b']:
            ev_cnts['non-specific'] += 1
    
    total_contexts = len(ev_dict)
    
    ev_counts_norm = dict()
    for ev, cnt in ev_cnts.items():
        ev_counts_norm[ev]  = cnt/total_contexts
    return ev_counts_norm


def get_evidence_diversity(model_name, model, prop, evidence_types, top_cutoff, concept_cutoff):
    
    # current file:
    annotation_name = f'annotation-tfidf-top_{top_cutoff}_{concept_cutoff}-raw-10000-categories'
    path_dir_annotation = f'../analysis/{model_name}/{annotation_name}/{prop}'
    f_annotation = f'{path_dir_annotation}/annotation-updated-done.csv'
    
    
    ev_dict = get_evidence_dict(model_name, prop, top_cutoff, concept_cutoff)
    
    
    ev_types_dict = defaultdict(list)
    
    for e, et in ev_dict.items():
        ev_types_dict[et].append(e)
        if et != 'u':
            ev_types_dict['all'].append(e)
        if et in ['p', 'l', 'n']:
            ev_types_dict['prop_specific'].append(e)
        elif et in ['i', 'r', 'b']:
            ev_types_dict['non-specific'].append(e)
    
    
    et_counts = dict()
    et_sims = dict()
    for et, words in ev_types_dict.items():
        if et in evidence_types:
            et_counts[et]  = len(words)
            mean_sim = get_mean_sim(model, words)
            et_sims[et] = mean_sim
    return et_counts, et_sims


def get_mean_sim(model, evidence_words):
    all_pairs = []
    for e1 in evidence_words:
        for e2 in evidence_words:
            pair = {e1, e2}
            if len(pair) ==2 and pair not in all_pairs:
                all_pairs.append(pair)


    all_similarities = []
    for pair in all_pairs:
        e1, e2 = list(pair)
        sim = model.similarity(e1, e2)
        all_similarities.append(sim)
    
    if len(all_similarities) > 0:
        mean_sim = sum(all_similarities)/len(all_similarities)
    else:
        mean_sim = np.nan
    return mean_sim