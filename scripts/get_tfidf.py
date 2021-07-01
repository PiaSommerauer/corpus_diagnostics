import json
import pandas as pd
from statistics import stdev
import os
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties
            
        

def load_data(prop):
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict

def combine_contexts(prop, label, target_words, model):
    
    c = ''
    for target in target_words:
        path = f'../contexts/{model}/vocab/{target}.txt'
        if os.path.isfile(path):
            #print('found file', target)
            with open(path) as infile:
                c += f' {infile.read()}'
        else:
            print('not found', target)
            print(path)
    path_dir = f'../contexts/{model}/{prop}/{label}'
    os.makedirs(path_dir, exist_ok=True)
    path_all = f'{path_dir}/ALL.txt'
    with open(path_all, 'w') as outfile:
        outfile.write(c)
        
def get_tfidf(path_target, paths_corpus, class_label, cnt_type='raw', max_features = 1000):
         
    # first path is pos, rest is neg:
    paths = [path_target]
    paths.extend(paths_corpus)
    if cnt_type == 'raw':
        vectorizer = tfidf(input = 'filename', max_features=max_features)
    elif cnt_type == 'binary':
        vectorizer = tfidf(input = 'filename', binary=True, max_features=max_features)
    x = vectorizer.fit_transform(paths)
    x = x.toarray()

    vocab = vectorizer.get_feature_names()


    # number_of_paths
    n_paths = len(paths)
    vec_target = x[0]

    vec_dict = dict()
    vec_dict['target'] = vec_target
    vec_dict['mean_corpus'] =  []
    for vec in x.T:
        vec_corpus = vec[1:]
        mean_corpus = sum(vec_corpus)/len(vec_corpus)
        vec_dict['mean_corpus'].append(mean_corpus)
    for n, path in enumerate(paths):
        if n > 0:
            vec_dict[path] = x[n]
    df = pd.DataFrame(vec_dict, index = vocab).sort_values('target', ascending=False)
    diff = []
    for t, corp in zip(df['target'], df['mean_corpus']):
        diff.append(t-corp)
    df['diff'] = diff
    return df


def tfidf_target_vs_corpus(prop, model, cnt_type, target_label, max_features):
    condition = 'each_target_vs_corpus'
    load_data(prop)

    path_results_dir = f'../results/{model}/tfidf-{cnt_type}-{max_features}/{condition}/{prop}/{target_label}'
    os.makedirs(path_results_dir, exist_ok=True)

    prop_dict = load_data(prop)
    target_pos = [c for c, d in prop_dict.items() if d['ml_label'] in ['all', 'all-some', 'few-some', 'some']]
    target_neg = [c for c, d in prop_dict.items() if d['ml_label'] in ['few']]

    if target_label == 'pos':
        target = target_pos
        corpus = target_neg
    elif target_label == 'neg':
        target = target_neg
        corpus = target_pos
    #print(len(target), len(corpus))
    
    # collect corpus paths
    paths_corpus = []
    for concept in corpus:
        path = f'../contexts/{model}/vocab/{concept}.txt'
        if os.path.isfile(path):
            #print('found path corpus', path)
            paths_corpus.append(path)
    
    for concept in target:
        path_target = f'../contexts/{model}/vocab/{concept}.txt'
        if os.path.isfile(path_target):
            #print('found path', path_target)
            #print('corpus paths:', len(paths_corpus))
            df = get_tfidf(path_target, paths_corpus, target_label, cnt_type, max_features)
            df = df[['target', 'mean_corpus', 'diff']]
            df.to_csv(f'{path_results_dir}/{concept}.csv') 
    

def main():
 
    model = 'giga_full'
    cnt_type = 'raw'
    max_features = 10000
    test = False
    properties_test = []
    labels = ['pos', 'neg']
    
    properties = get_properties()
    #properties_test = ['square', 'dangerous']
    #properties = [p for p in properties if p not in properties_test]
    
    for prop in properties:
        print(prop)
        for target_label in labels:
            print(target_label)
            tfidf_target_vs_corpus(prop, model, cnt_type, target_label, max_features)

            
            
if __name__ == '__main__':
    main()
    
    
       