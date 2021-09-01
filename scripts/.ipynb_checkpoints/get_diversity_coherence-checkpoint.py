import analyze_annotations
import pandas as pd
from gensim.models import KeyedVectors



path_giga = '/Users/piasommerauer/Data/DSM/corpus_exploration/giga_full/sgns_pinit1/sgns_rand_pinit1.words'
path_wiki = '/Users/piasommerauer/Data/DSM/corpus_exploration/wiki_full/trained_for_analysis_June2021/sgns_pinit1/sgns_rand_pinit1.words'


model_name = 'wiki_updated'
model_path  = path_wiki

# load model
if model_name == 'googlenews':
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
else:
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)

top_cutoff = 3
concept_cutoff = 3
ann_dict = analyze_annotations.show_annotation_status(model_name, top_cutoff, concept_cutoff)
properties = ann_dict['complete']


#cols = ['property', 'u', 'all', 'prop_specific', 'non-specific'] # 'p', 'n', 'l', 'i', 'r', 'b']
cols = ['property']
e_types = ['all', 'prop_specific', 'non-specific']
#e_types = ['prop_specific']

for et in e_types:
    cols.append(f'{et}-div')
    cols.append(f'{et}-sim')
print(cols)
    
table = []
for prop in list(properties):
    d = dict()
    d['property'] =  prop
    et_counts, et_sims = analyze_annotations.get_evidence_diversity(model_name, model, 
                                                                    prop, e_types, top_cutoff, concept_cutoff)
    for et, count in et_counts.items():
        sim = et_sims[et]
        d[f'{et}-div'] = count
        d[f'{et}-sim'] = sim
    #d.update(analyze_annotations.get_evidence_diversity(model_name, prop, top_cutoff, concept_cutoff))
    for c in cols:
        if c not in d:
            d[c] = 0#np.nan
    table.append(d)
   
#cols = ['property', 'u', 'all', 'prop_specific', 'non-specific'] #, 'p', 'n', 'l', 'i', 'r', 'b']
df = pd.DataFrame(table)[cols]
df = df[cols].sort_values('prop_specific-div', ascending = False) #.round(2) #.round(0)
#print(df.to_latex(index=False))
df.to_csv(f'../analysis/diversity-coherence-{model_name}.csv')