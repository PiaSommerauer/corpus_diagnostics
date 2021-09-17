import sys
import analyze_annotations
import pandas as pd



model_name = sys.argv[1]
#model_name = 'giga_full_updated'

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
print(cols)
    
table = []
for prop in list(properties):
    d = dict()
    d['property'] =  prop
    et_counts = analyze_annotations.get_evidence_diversity(model_name, prop, e_types, top_cutoff, concept_cutoff)
    for et, count in et_counts.items():
        d[f'{et}-div'] = count
    #d.update(analyze_annotations.get_evidence_diversity(model_name, prop, top_cutoff, concept_cutoff))
    for c in cols:
        if c not in d:
            d[c] = 0#np.nan
    table.append(d)
   
#cols = ['property', 'u', 'all', 'prop_specific', 'non-specific'] #, 'p', 'n', 'l', 'i', 'r', 'b']
df = pd.DataFrame(table)[cols]
df = df[cols].sort_values('prop_specific-div', ascending = False) #.round(2) #.round(0)
#print(df.to_latex(index=False))
df.to_csv(f'../analysis/diversity-{model_name}.csv')