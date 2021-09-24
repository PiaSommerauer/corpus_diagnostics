from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import csv
import os

from gensim.models import KeyedVectors

import utils
import relations


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


def get_evidence_sim(evidence_type_dict, model):
    
    evidence_sim_dict = dict()
    type_evidence_dict = defaultdict(list)
    
    for c, t in evidence_type_dict.items():
        #type_evidence_dict[t].append(c)
        if t in ['p', 'n', 'l']:
            t_c = 'prop-specific'
            type_evidence_dict[t_c].append(c)
        elif t in ['i', 'r', 'b']:
            t_c = 'non-specific'
            type_evidence_dict[t_c].append(c)
        elif t == 'u':
            t_c = 'u'
            type_evidence_dict[t_c].append(c)
                 
    for t, contexts in type_evidence_dict.items():
        print(t, len(contexts))
        mean_sim = get_mean_sim(model, contexts)
        evidence_sim_dict[t] = mean_sim
        
    return evidence_sim_dict


def get_evidence_sim_properties(model_name, model):
    
    table = []
    
    properties = utils.get_properties()

    for prop in properties:
        print('starting property', prop)
        evidence_type_dict = utils.load_evidence_type_dict(prop, model_name)
        evidence_sim = get_evidence_sim(evidence_type_dict, model)
        evidence_sim['property'] = prop
        table.append(evidence_sim)
        print('finished prop', prop)
    
    #columns = ['prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    columns = ['prop-specific', 'non-specific', 'u']
    df = pd.DataFrame(table).set_index('property')[columns]
    # set nana to 0 before median
    #df = df.fillna(0.0)
    median = df.median(axis=0)
    df.loc['median'] = median
    
    return df


def get_evidence_sim_concept_category(model_name, evidence_type_dict, 
                                           prop, concept, label, category, model):
    
    evidence_sim_dict = dict()

    
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    full_path = f'{dir_path}/{prop}/{category}/{label}/{concept}.csv'
    
    with open(full_path) as infile:
        data = list(csv.DictReader(infile))
    contexts = [d[''] for d  in data if float(d['diff']) > 0]
    
    context_candidates = [c for c in contexts if c in evidence_type_dict.keys()]
    n_candidates = len(context_candidates)
    
    evidence_context_dict = defaultdict(list)
    for c in context_candidates:
        if c in evidence_type_dict:
            t = evidence_type_dict[c]
            #evidence_context_dict[t].append(c)
            if t in ['p', 'n', 'l']:
                t_c = 'prop-specific'
                evidence_context_dict[t_c].append(c)
            #too much computing time:
#             elif t in ['i', 'r', 'b']:
#                 t_c = 'non-specific'
#                 evidence_context_dict[t_c].append(c)
    for t, contexts in evidence_context_dict.items():
        #n_contexts = len(contexts)
        mean_sim = get_mean_sim(model, contexts)
        evidence_sim_dict[t] = mean_sim
    
    return evidence_sim_dict


def get_evidence_sim_concept(prop, concept, label, 
                                  evidence_type_dict, model_name, model):
    categories = utils.get_categories_concept(prop, concept, model_name)
    ev_sim_concept = Counter()

    for cat in categories:
        ev_sim_concept_cat = get_evidence_sim_concept_category(model_name, evidence_type_dict, 
                                                                     prop, concept, label,
                                                                     cat, model)

        for ev, sim in ev_sim_concept_cat.items():
            ev_sim_concept[ev] += sim

    # calculate means
    for ev, sim in ev_sim_concept.items():
        sim_mean = sim/len(categories)
        ev_sim_concept[ev] = sim_mean
    return ev_sim_concept 



def get_evidence_sim_concepts(model_name, properties, model):
    
    table = []
    keys = set()
    
    for prop in properties:
        concept_label_dict = dict()
        concepts_pos = utils.get_examples(model_name, prop, 'pos')
        concepts_neg = utils.get_examples(model_name, prop, 'neg')
        for c in concepts_pos:
            concept_label_dict[c] = 'pos'
        for c in concepts_neg:
            concept_label_dict[c] = 'neg'
        evidence_type_dict = utils.load_evidence_type_dict(prop, model_name)

        for concept, label in concept_label_dict.items():
            ev_sim_concept =  get_evidence_sim_concept(prop, concept, label,
                                                             evidence_type_dict, 
                                                             model_name, model)
            ev_sim_concept['label'] = label
            keys.update(ev_sim_concept.keys())
            ev_sim_concept['pair'] = (prop, concept)
            table.append(ev_sim_concept)
        print('finished property:', prop)
        
    columns = ['label', 'prop-specific']#, 'non-specific'], 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    columns = [c for c in columns if c in keys]
    df = pd.DataFrame(table).set_index('pair')
    # do not set to 0
    median = df.median(axis=0)
    df.loc['median'] = median
    df = df[columns]
    return df


def main():
    
    
    
    model_names = ['giga_full_updated', 'wiki_updated']
    analysis_name = 'coherence'
    
    path_giga = '/Users/piasommerauer/Data/DSM/corpus_exploration/giga_full/sgns_pinit1/sgns_rand_pinit1.words'
    path_wiki = '/Users/piasommerauer/Data/DSM/corpus_exploration/wiki_full/trained_for_analysis_June2021/sgns_pinit1/sgns_rand_pinit1.words'
  
    properties = utils.get_properties() 
    
    for model_name in model_names:
        
        # find model path
        if model_name == 'giga_full_updated':
            model_path = path_giga
        else:
            model_path  = path_wiki

        # load model
      
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        print('loaded model', model_name)
#         #### prop
        

        # properties
        level = 'properties'
        df = get_evidence_sim_properties(model_name, model)
        # to file
        path_dir = f'../analysis/{model_name}/{level}/'
        os.makedirs(path_dir, exist_ok=True)
        path_file = f'{path_dir}/{analysis_name}.csv'
        df.to_csv(path_file)
        print('analyzed', level)
        
#         #pairs
#         level = 'pairs'
#         df = get_evidence_sim_concepts(model_name, properties, model)
#         # to file
#         path_dir = f'../analysis/{model_name}/pairs/'
#         os.makedirs(path_dir, exist_ok=True)
#         path_file = f'{path_dir}/{analysis_name}.csv'
#         df.to_csv(path_file)
#         print('finished analysis:', level)

#         # relations
#         level = 'relations'
#         pair_score_dict = relations.load_scores(analysis_name, model_name)
#         df = relations.relation_overview(pair_score_dict)
#         # to file:
#         path_dir = f'../analysis/{model_name}/{level}/'
#         os.makedirs(path_dir, exist_ok=True)
#         path_file = f'{path_dir}/{analysis_name}.csv'
#         df.to_csv(path_file)
       
        
        
if __name__ == '__main__':
    main()