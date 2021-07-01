import os
import csv
from collections import Counter, defaultdict
import json
import pandas as pd


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties


def get_example_data(prop, class_label):
    concept_rel_dict = dict()
    path = f'../data/aggregated_semantic_info/{prop}.json'
    if label_class == 'pos':
        target_labels = ['all', 'all-some', 'some','few', 'some-few']
    elif label_class == 'neg':
        target_labels = ['few']
    with open(path) as infile:
        data = json.load(infile)
    for concept, d in data.items():
        label = d['ml_label']
        if label in targt_labels:
            concept_rel_dict[concept]=d['relations']
    return concept_rel_dict

def get_tfidf_data(prop, cnt_type, max_features, class_label, model):
    path = f'../results/{model}/tfidf-{cnt_type}-{max_features}/each_target_vs_corpus/{prop}/{class_label}'
    concept_tfidf_dict = dict()
    concept_diff_dict = dict()
    for f in os.listdir(path):
        if f.endswith('.csv'):
            pos_concept = f.split('.')[0]
            full_path = f'{path}/{f}'
            #print(full_path)
            with open(full_path) as infile:
                data = list(csv.DictReader(infile))
            context_tfidf_dict = Counter()
            context_diff_dict = dict()
            for d in data:
                context_tfidf_dict[d['']] = float(d['target'])
                context_diff_dict[d['']] = float(d['diff'])
            concept_tfidf_dict[pos_concept] = context_tfidf_dict
            concept_diff_dict[pos_concept] = context_diff_dict
    return concept_tfidf_dict, concept_diff_dict

def get_context_counts(concept_tfidf_dict, concept_diff_dict, concepts_category):
    context_counter = Counter()
    context_value_dict = dict()
    context_differences_dict = dict()
    for concept, context_tfidf_dict in concept_tfidf_dict.items():
        if concept in concepts_category:
            context_diff_dict = concept_diff_dict[concept]
            for context, tfidf in context_tfidf_dict.most_common():
                # check if tfidf higher than mean neg:
                difference = context_diff_dict[context]
                if difference > 0:
                    context_counter[context] += 1
                    if context in context_value_dict:
                        context_value_dict[context][concept] = tfidf
                        context_differences_dict[context][concept] = difference
                    else:
                        context_value_dict[context] = dict()
                        context_differences_dict[context] = dict()
                        context_value_dict[context][concept] = tfidf
                        context_differences_dict[context][concept] = difference

    return context_counter, context_value_dict, context_differences_dict

def to_file(prop, category, context_counter, 
            context_value_dict, context_differences_dict, 
            model, cnt_type, max_features, class_label, concepts):
  
    dir_path_neg = f'../analysis/{model}/tfidf_aggregated_concept_scores-{cnt_type}-{max_features}/'
    dir_path_pos = f'../analysis/{model}/tfidf_aggregated_concept_scores-{cnt_type}-{max_features}/{prop}-pos/{category}/'
    path_pos = f'{dir_path_pos}{prop}-pos.csv'
    path_neg = f'{dir_path_neg}{prop}-neg.csv'
    if class_label == 'pos':
        dir_path = dir_path_pos
        full_path = path_pos
    elif class_label == 'neg':
        dir_path = dir_path_neg
        full_path = path_neg
    os.makedirs(dir_path, exist_ok=True)
    #n_concepts = len(concept_tfidf_dict)
    with open(f'{dir_path}{prop}-{class_label}-config.txt', 'w') as outfile:
        outfile.write(prop+'\n')
        outfile.write(f'n pos examples: {len(concepts)}\n')
        outfile.write(f'cut-off: difference above 0\n')
        outfile.write(f'max_features: {max_features}')
    with open(f'{dir_path}{prop}-{class_label}-concepts.txt', 'w') as outfile:
        outfile.write('\n'.join(concepts))
    with open(full_path, 'w') as outfile:
        outfile.write('context,n_concepts,mean_tfidf,mean_diff,min_tfidf,max_tfidf\n')
        for c, cnt in context_counter.most_common():
            #print(c, cnt)
            values = context_value_dict[c].values()
            diffs = context_differences_dict[c].values()
            mean = sum(values)/len(values)
            mean_diff = sum(diffs)/len(diffs)
            line = [c, str(cnt), str(round(mean, 2)), str(round(mean_diff, 2)),
                    str(round(min(values), 2)), str(round(max(values), 2))]
            outfile.write(','.join(line)+'\n')
            

    
    
def tfidf_to_file(prop, model, cnt_type, max_features, class_label, cat_concepts_dict):

    concept_tfidf_dict, concept_diff_dict = get_tfidf_data(prop, cnt_type, max_features, class_label, model)
    if class_label == 'pos':
        for category, concepts_category in cat_concepts_dict.items():
            if category != 'all-neg':
                context_counter, context_value_dict, context_differences_dict = get_context_counts(
                                                                            concept_tfidf_dict, 
                                                                            concept_diff_dict,
                                                                            concepts_category)
                to_file(prop, category, context_counter, 
                     context_value_dict, context_differences_dict, 
                        model, cnt_type, max_features, class_label, concepts_category)
    else:
        category = 'all-neg'
        concepts_category = cat_concepts_dict[category]
        context_counter, context_value_dict, context_differences_dict = get_context_counts(
                                                                            concept_tfidf_dict, 
                                                                            concept_diff_dict,
                                                                            concepts_category)
        to_file(prop, category, context_counter, 
                     context_value_dict, context_differences_dict, 
                        model, cnt_type, max_features, class_label, concepts_category)

        
        
def get_table(model, prop, category, cnt_type, max_features, rank_by='n_concepts', top_n=20):
    cols = ['context', 'n_concepts', 'mean_tfidf', 'mean_diff']
    dir_path_neg = f'../analysis/{model}/tfidf_aggregated_concept_scores-{cnt_type}-{max_features}/'
    dir_path_pos = f'../analysis/{model}/tfidf_aggregated_concept_scores-{cnt_type}-{max_features}/{prop}-pos/{category}/'
    path_pos = f'{dir_path_pos}{prop}-pos.csv'
    path_neg = f'{dir_path_neg}{prop}-neg.csv'
    
    prop_word_dict = {'lay_eggs': {'eggs', 'egg'}, 
                       'made_of_wood': {'wood', 'wooden'},
                       'used_in_cooking': {'cooking', 'cook', 'cooks', 'cooked'},
                       'swim' : {'swim', 'swam', 'swum', 'swimming', 'swims'},
                       'fly': {'flying', 'fly', 'flew', 'flown', 'flies'},
                       'roll': {'rolling', 'roll', 'rolled', 'rolls'},
                        'wheels': {'wheel', 'wheels'},
                        'wings' : {'wing','wings'}
                         }
    print(path_pos)
    df_pos = pd.read_csv(path_pos)
    df_neg = pd.read_csv(path_neg)
    #target_prop_line
    rows_pos = []
    if prop in prop_word_dict:
        prop_evidence = prop_word_dict[prop]
    else:
        prop_evidence = {prop}
    for i, row in df_pos.iterrows():
        if row['context'] in prop_evidence:
            rows_pos.append(row)
    rows_neg = []
    for i, row in df_neg.iterrows():
        if row['context'] in prop_evidence:
            rows_neg.append(row)

    df_pos = df_pos.sort_values(rank_by, ascending=False)[:top_n]
    df_neg = df_neg.sort_values(rank_by, ascending=False)[:top_n]
    
    for row in rows_pos:
        df_pos = df_pos.append(row, ignore_index=True)
    for row in rows_neg:
        df_neg = df_neg.append(row, ignore_index=True)
    
    return df_pos[cols], df_neg[cols]


def tables_to_file(model, prop, cnt_type, max_features, cat_concepts_dict):
    
    for category in cat_concepts_dict:
        if category != 'all-neg':
            dir_path_pos = f'../analysis/{model}/annotation-tfidf-top20-{cnt_type}-{max_features}/{prop}-pos/{category}/'
            path_pos = f'{dir_path_pos}{prop}-pos.csv'
            os.makedirs(dir_path_pos, exist_ok=True)
            df_pos, df_neg = get_table(model, prop, category, cnt_type, max_features,
                                       rank_by='n_concepts', top_n=20)
            evidence = ['-' for n in range(len(df_pos))]
            df_pos['evidence'] = evidence
            df_pos.to_csv(path_pos)

    dir_path_neg = f'../analysis/{model}/annotation-tfidf-top20-{cnt_type}-{max_features}/'
    path_neg = f'{dir_path_neg}{prop}-neg.csv'
    os.makedirs(dir_path_neg, exist_ok=True)
    df_neg.to_csv(path_neg)
        
        
def get_cat_concepts(prop):
    path = f'../data/aggregated_semantic_info/{prop}.json'
    valid_labels = ['all', 'all-some', 'some','few']
    labels_pos = ['all', 'all-some', 'some', 'some-few']
    labels_neg = ['few']
    
    cat_concept_dict = defaultdict(set)
    
    with open(path) as infile:
        concept_dict = json.load(infile)
        
    for concept, d in concept_dict.items():
        categories_dict = d['categories']
        label = d['ml_label']
        if label in valid_labels:
            if label in labels_pos:
                cat_concept_dict['all-pos'].add(concept)
                for cat in categories_dict.keys():
                    cat_concept_dict[cat].add(concept)
            elif label in labels_neg:
                cat_concept_dict['all-neg'].add(concept)
    # remove categories smaller than 5:
    cat_concept_dict_clean = defaultdict(set)
    concepts_in_large_cats = set()
    for cat, concepts in cat_concept_dict.items():
        if len(concepts) > 5:
            concepts_in_large_cats.update(concepts)
    for cat, concepts in cat_concept_dict.items():
        if len(concepts) <= 5:
            concepts_in_other_cats = [c in concepts_in_large_cats for c in concepts]
            if not all(concepts_in_other_cats):
                cat = 'fly'
            else:
                # concepts already in other categories
                continue
        cat_concept_dict_clean[cat].update(concepts) 
    # merge equivalent categories:
    cat_concepts_merged = defaultdict(set)
    for cat1, concepts1 in cat_concept_dict_clean.items():
        for cat2, concepts2 in cat_concept_dict_clean.items():
            # check if different categories are equivalent:
            if cat1 != cat2:
                if concepts1 == concepts2:
                    # found equivalent cats
                    cat = '-'.join(sorted([cat1, cat2]))
                    cat_concepts_merged[cat] = concepts1
                else:
                    cat_concepts_merged[cat1] = concepts1
                    cat_concepts_merged[cat2] = concepts2
    return cat_concepts_merged
        
        
        
        
def main():
    
    cnt_type = 'raw'
    max_features = 10000
    model = 'giga_full'
    labels = ['pos', 'neg']
    #properties = ['square', 'dangerous']
    properties = get_properties()
    

    for prop in properties:
        cat_concept_dict = get_cat_concepts(prop)
        for class_label in labels:
            tfidf_to_file(prop, model, cnt_type, max_features,
                          class_label, cat_concept_dict)
        tables_to_file(model, prop, cnt_type, max_features, cat_concept_dict)
        
        
if __name__ == '__main__':
    main()