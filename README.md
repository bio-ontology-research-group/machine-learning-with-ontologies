# Materials for Machine Learning with Ontologies

This repository contains all the materials for our "Machine learning with biomedical ontologies" manuscript. We provide the Jupyter Notebooks to reproduce our experimental results and the benchmark datasets based on predicting protein-protein interactions.
Furthermore, we make a set of slides available (as PDF and source code in LaTeX Beamer) that may be useful for teaching or presentations.

## Notebooks

We provide several [Jupyter](https://jupyter.org/) notebooks. The notebooks include:

* [data.ipynb](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/data.ipynb) -- data generation and preprocessing
* [semantic-similarity.ipynb](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/semantic-similarity.ipynb) -- predicting protein--protein interactions using semantic similarity measures
* [onto2vec_opa2vec.ipynb](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/onto2vec_opa2vec.ipynb) -- axiom embeddings using Onto2Vec and embeddings for axioms plus annotation properties using OPA2Vec
* [graph-walks.ipynb](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/graph-walks.ipynb) -- embedding ontologies using random walks on graphs
* [graph-transe.ipynb](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/graph-transe.ipynb) -- embedding ontologies using translating embeddings on graphs
* [elembedding.ipynb](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/elembedding.ipynb) -- ontology embedding that approximates the ontology models (EL Embeddings)


## PPI Benchmark

We provide two benchmark datasets for protein--protein interaction
prediction task. The datasets can be downloaded using the following
link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3779900.svg)](https://doi.org/10.5281/zenodo.3779900)

Two benchmark datasets for evaluating machine learning methods on the
task of predicting protein--protein interaction networks. The original
data was downloaded from StringDB database of protein--protein
interactions and Gene Ontology Resource. This archive includes:

* Protein--protein interactions for human and yeast organisms
* Gene Ontology in OBO and OWL format
* Gene Ontology Annotations for human and yeast proteins
* Protein aliases files with ID mappings between StringDB proteins and
  other databases.
  
We filter out interactions with confidence score less than 700 and
consider them to be symmetric. We randomly split the datasets into
80/20% training/testing sets by the number of interactions and use 20% of the training set as a validation set.

### Dependencies

Please install the following software to run our notebooks:
* [Groovy](https://groovy-lang.org/install.html) 2.0+
* [Raptor RDF Syntax Library](http://librdf.org/raptor/)
* Python 3.6+
* Install python dependencies with: ```pip install -r requirements.txt```
* Load submodules with: ```git submodule update --init --recursive```

### Running the notebooks
Run `jupyter notebook` and then open the notebook files.

### Current benchmark results (yeast)

| Method           |Raw Hits\@10  |
|------------------|--------------|
| TransE           |0.06          |
| SimResnik        |**0.09**      |
| SimLin           |0.08          |
| SiameseNN        |0.06          |
| SiameseNN (Ont)  |0.08          |
| EL Embeddings    |0.08          |
| Onto2Vec         |0.08          |
| OPA2Vec          |0.06          |
| Random walk      |0.06          |
| Node2Vec         |0.07          |


### Current benchmark results (human)

### Adding to the benchmark
To add your own results to the benchmark, please send us a pull request with a link to the source repository that contains the code to reproduce the results.

## Slides

We provides slides that can be used to present some of this work. The slides have been created as part of an [Ontology Tutorial](https://github.com/bio-ontology-research-group/ontology-tutorial) that was developed and taught over several years at various events.
All methods in the slides are also implemented with examples in our Jupyter Notebooks.

1. [Introduction](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/slides/01-introduction.pdf)
2. [Ontologies and Graphs](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/slides/02-ontologies-and-graphs.pdf) -- basic introduction to ontologies, Description Logic, and how they can give rise to graph-based representations
3. [Semantic Similarity](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/slides/03-semantic-similarity.pdf) -- different semantic similarity measures on ontologies
4. [Ontology Embeddings](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies/blob/master/slides/04-ontology-embeddings.pdf) -- methods to generate embeddings for ontologies, including syntactic, graph-based, and model-based approaches.

## Resources

### Processing and pre-processing ontologies

* [OWLAPI](https://github.com/owlcs/owlapi): Reference library to process OWL ontologies, supports most OWL reasoners.
* [funowl](https://github.com/hsolbrig/funowl): Python library to process OWL ontologies.
* [owlready2](https://pypi.org/project/Owlready2/): Python library to process OWL ontologies.
* [Apache Jena](https://jena.apache.org/): RDF library with OWL support.
* [rdflib](https://github.com/RDFLib/rdflib): Python RDF library with OWL support, in particular infixowl.
* [Protege](https://protege.stanford.edu/): Comprehensive ontology editor and knowledge engineering environment.

### Computing entailments, reasoning

* [ELK](https://github.com/liveontologies/elk-reasoner): Very fast reasoner for the OWL 2 EL profile with polynomial worst-case time complexity.
* [HermiT](http://www.hermit-reasoner.com/): Automated reasoner supporting most of OWL axioms with exponential worst-case complexity.
* [Pellet](https://github.com/stardog-union/pellet): OWL reasoner supporting most of the OWL constructs and supporting several additional features.

### Generating graphs from ontologies

* [OBOGraphs](https://github.com/geneontology/obographs): Syntactic conversion of ontologies to graphs, targeted at OBO ontologies.
* [Onto2Graph](https://github.com/bio-ontology-research-group/Onto2Graph): Semantic conversion of OWL ontologies to graphs using automated reasoning, following the axiom patterns of the OBO Relation Ontology.

### Computing Semantic Similarity

* [Semantic Measures Library](http://www.semantic-measures-library.org/sml/): Comprehensive Java library to compute semantic similarity measures over ontologies.
* [sematch](https://github.com/gsi-upm/sematch): Python library to compute semantic similarity on knowledge graphs.
* [DiShIn](https://github.com/lasigeBioTM/DiShIn): Python library to compute semantic similarity on ontologies.

### Embedding graphs

* [OWL2Vec](https://github.com/oholter/matcher-with-word-embedings): Graph-based ontology embedding method that combines generation of graphs from ontologies, random walks on the generated graphs, and generation of embeddings using Word2Vec. Supports most OWL axioms.
* [DL2Vec](https://github.com/bio-ontology-research-group/DL2Vec): Graph-based ontology embedding method that combines generation of graphs from ontologies, random walks on the generated graphs, and generation of embeddings using Word2Vec. Supports most OWL axioms.
* [Walking RDF & OWL](https://github.com/bio-ontology-research-group/walking-rdf-and-owl): Graph-based ontology embedding method that combines generation of graphs from ontologies, random walks on the generated graphs, and generation of embeddings using Word2Vec. Supports only the ontology taxonomy.
* [RDF2Vec](https://github.com/dwslab/jRDF2Vec) ([pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec)): Method to embed RDF graphs. 
* [Node2Vec](http://snap.stanford.edu/node2vec/): Method to embed graphs using biased random walks.
* [PyKEEN](https://github.com/SmartDataAnalytics/PyKEEN) and [BioKEEN](https://github.com/SmartDataAnalytics/BioKEEN): Toolkit for generating knowledge graph embeddings using several different approaches. 
* [OpenKE](https://github.com/thunlp/OpenKE): Library and toolkit for generating knowledge graph embeddings
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric): Library for graph neural networks which can be used to generate graph embeddings.


### Embedding axioms

* [Onto2Vec](https://github.com/bio-ontology-research-group/onto2vec): Embeddings based on treating logical axioms as a text corpus.
* [OPA2Vec](https://github.com/bio-ontology-research-group/opa2vec): Embeddings that combine logical axioms with annotation properties.
* [EL Embeddings](https://github.com/bio-ontology-research-group/el-embeddings): Embeddings that approximate the interpretation function and preserve semantics for intersection, existential quantifiers, and bottom class.

### Ontology-based constrained learning:

* [DeepGO](https://github.com/bio-ontology-research-group/deepgo) and
  [DeepPheno](https://github.com/bio-ontology-research-group/deeppheno):
  Implement ontology-based hierarchical classifiers for function and
  phenotype prediction.  The hierarchical classification modules are
  generic and can be used with other ontologies and applications.


## Publication

If you like our work, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2020.05.07.082164v1).
