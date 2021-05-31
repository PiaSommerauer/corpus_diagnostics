# From corpus to pairs

Ideally, we use a standardized way to pre-process the corpora

* wiki
* giga


This script processes wiki using stanford core nlp: https://github.com/jind11/word2vec-on-wikipedia

The models in the nlpl repository also use stanford nlp for pre-processing. 

This seems reasonable to me - I think we should use this for both, wiki and giga.

By using the same process, we make the results more comparable. We can provide the scripts in with the code of our full experiments. 


Installation of stanfrod nlp:

Go to https://stanfordnlp.github.io/CoreNLP/index.html#download

* Download stanford core nlp + models for English. 
* Move models into stanford core nlp repo
* Add everything to path as indicated in the instructions. 

Follow word2vec-on-wikipedia instructions:

* Set up server in stanford-nlp directory
* install python wrapper: https://github.com/smilli/py-corenlp


