from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import csv
import os
import itertools

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import utils
import relations


def get_mean_sim(model, evidence_words):
    all_pairs = []
    words1 = []
    words2 = []
    
    # check if evidence words in model vocab:
    evidence_words = [w for w in evidence_words if w in model.vocab]
    
    
    print('creating pairs')
    pairs = list(itertools.combinations(evidence_words, 2))
    print('created pairs', len(pairs))
    
    
    n_pairs = len(pairs)
    size_batches = 10000
    if n_pairs > size_batches:
        n_batches = int(n_pairs/size_batches)
    else:
        n_batches = 1

    batch_ints= []
    previous_end = 0
    for n in range(n_batches):
        start = previous_end 
        end = start+(size_batches)
        previous_end = end
        batch_ints.append((start, end))
    if previous_end < n_pairs:
        batch_ints.append((previous_end, n_pairs))

    
    cos_sum = 0.0
    print('starting batches for pairs:',   n_batches)
    for start, end in batch_ints:
        print('load vecs')
        words1 = [model[w1] for w1, w2 in pairs[start:end]]
        words2 = [model[w2] for w1, w2 in pairs[start:end]]
        print('loaded vecs')

        # make two matrices:


        words1 = np.array(words1)
        words2 = np.array(words2)

        print(words1.shape)
        print(words2.shape)

        print('calculating cosine')

        if len(words1) > 0 and len(words2) > 0:
            results = cosine_similarity(words1, words2)
            i = 0
            for v in results:
                #print(v, i)
                cos = v[i]
                i += 1
                cos_sum+= cos
        else:
            cos_sum += 0
    print('finished batches')
    print('calculating mean', cos_sum, n_pairs)
    
    if cos_sum > 0.0:
        cos_mean = cos_sum/n_pairs
    else:
        cos_mean = np.nan
#         if len(evidence_words) == 1:
#             cos_mean = 1.0
#         else:
#             cos_mean = 0.0
        
    print('finished calculating cosine')


    return cos_mean


def get_evidence_sim(evidence_type_dict, model):
    
    evidence_sim_dict = dict()
    type_evidence_dict = defaultdict(list)
    
    for c, t in evidence_type_dict.items():
        #type_evidence_dict[t].append(c)
        t_c = 'all'
        type_evidence_dict[t_c].append(c)
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
    columns = ['all', 'prop-specific', 'non-specific', 'u', 'all']
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