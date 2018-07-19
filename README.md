## ELMoLex - Connecting ELMo and Lexicon features for Dependency Parsing
Our [CONLL-18 submission system](https://drive.google.com/open?id=1zD4Fa5OaL7YuxZNU7O-MhqyRKvFZ0H6s) for [Parsing](http://universaldependencies.org/conll18/) which uses [Deep Biaffine Parser](https://arxiv.org/abs/1611.01734) as its backbone.

### Features
* [ELMo](https://allennlp.org/elmo) features
* [UDLexicons](http://pauillac.inria.fr/~sagot/index.html#udlexicons) features
* LSTM pre-training through NLM (NLM Init. feature)
* Read/Write directly in CONLL-U
* Integration with CONLL-18 evaluation script

### Quick Start
* Install the dependencies (ensure conda is working before running the setup):
```
bash setup.sh
```
* Check if everything works through a dry run (should take less than 30 minutes):
```
bash sample_run/run.sh check
```
* Perform a complete run of the parser on en_lines treebank:
```
bash sample_run/run.sh complete
```

### Resources
* [Raw Treebanks](http://universaldependencies.org/conll18/data.html)
* [UDLexicons](http://pauillac.inria.fr/~sagot/index.html#udlexicons)
* [FastText Word Embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

### Acknowledgements
* [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2) from XuezheMax
* [Chu-Liu-Edmonds](https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/mst/mst.ipynb) from bastings

### License
ELMoLex is GPL-licensed.

