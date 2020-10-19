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

|  Method           | Raw Hits\@10 |  Filtered Hits\@10|   Raw Hits\@10 | Filtered Hits\@100 | Raw Mean Rank | Filtered Mean Rank |  Raw AUC |   Filtered AUC|
|-------------------|--------------|-------------------|----------------|--------------------|---------------|--------------------|----------|---------------|
|  TransE           | 0.06         |  0.13             |   0.32         |  0.40              |   1125.4      |    1074.8          |     0.82 |      0.83 |	
|  SimResnik        | **0.09**     |  0.17             |   0.38         |  0.48              |   757.8       |    706.9           |     0.86 |      0.87 |
|  SimLin           | 0.08         |  0.15             |   0.33         |  0.41              |   875.4       |    824.5           |     0.84 |      0.85|
|  SiameseNN        | 0.06         |  0.17             |   0.46         |  0.68              |   674.27      |    622.20          |     0.89 |      0.90|
|  SiameseNN (Ont)  | 0.08         |  **0.19**         |   **0.50**     |  **0.72**          |   543.56      |    491.56          |     0.91 |      0.92|
|  EL Embeddings    | 0.08         |  0.17             |   0.44         |  0.62              |   **451.29**  |    **394.04**      |     **0.92** |  **0.93**|
|  Onto2Vec         | 0.08         |  0.15             |   0.35         |  0.48              |   641.1       |    587.9           |     0.79 |      0.80|
|  OPA2Vec          | 0.06         |  0.13             |   0.39         |  0.58              |   523.3       |    466.6           |     0.87 |      0.88|
|  Random walk      | 0.06         |  0.13             |   0.31         |  0.40              |   612.6       |    587.4           |     0.87 |      0.88|
|  Node2Vec         | 0.07         |  0.15             |   0.36         |  0.46              |   589.1       |    522.4           |     0.87 |      0.88|


### Current benchmark results (human)

|Method           | Raw Hits\@10  | Filtered Hits\@10 | Raw Hits\@10  | Filtered Hits\@100 | Raw Mean Rank | Filtered Mean Rank | Raw AUC  |  Filtered AUC|
|-----------------| --------------|-------------------|---------------|--------------------|---------------|--------------------|----------|--------------|
|  TransE       |     **0.05**  |     0.11       |         0.24      |     0.29          |       3960.4    |      3890.6        |       0.78   |    0.79|
|  SimResnik     |    **0.05**  |     0.09       |         0.25      |     0.30          |       1933.6    |      1864.4        |       0.88   |    0.89|
|  SimLin        |    0.04      |     0.08       |         0.20      |     0.23          |       2287.9    |      2218.7        |       0.86   |    0.87|
|  SiameseNN      |   0.05      |     **0.15**   |         **0.41**  |     **0.64**      |       1881.10   |      1808.77       |       0.90   |    0.89|
|  SiameseNN (Ont)|   0.05      |     0.13       |         0.38     |      0.59          |       1838.31   |      1766.34       |       0.89   |    0.89|
|  EL Embeddings  |   0.01      |     0.02       |         0.22     |      0.26          |       **1679.72**|     **1637.65**   |       **0.90**|   **0.90**|
|  Onto2Vec       |   **0.05**  |     0.08       |         0.24     |      0.31          |       2434.6    |      2391.2        |       0.77    |   0.77|
|  OPA2Vec        |   0.03      |     0.07       |         0.23     |      0.26          |       1809.7    |      1767.6        |       0.86    |   0.88|
|  Random walk    |   0.04      |     0.10       |         0.28     |      0.34          |       1942.6    |      1958.6        |       0.85    |   0.86|
|  Node2Vec       |   0.03      |     0.07       |         0.22     |      0.28          |       1860.5    |      1813.1        |       0.86    |   0.87|

### Adding to the benchmark
To add your own results to the benchmark, please send us a pull request with a link to the source repository that contains the code to reproduce the results. Alternatively, please create an issue on the issue tracker and we will add your results.

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
* [OWL2Vec*](https://github.com/KRR-Oxford/OWL2Vec-Star): Graph-based ontology embedding method that combines generation of graphs from ontologies, random walks on the generated graphs, and generation of embeddings using Word2Vec. Supports OWL 2 axioms and annotation axioms. Extension of OWL2Vec.


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

If you like our work, please cite our [paper](https://doi.org/10.1093/bib/bbaa199):
```
@article{machine-learning-with-ontologies,
    author = {Kulmanov, Maxat and Smaili, Fatima Zohra and Gao, Xin and Hoehndorf, Robert},
    title = {Semantic similarity and machine learning with ontologies},
    journal = {Briefings in Bioinformatics},
    year = {2020},
    month = {10},
    abstract = {Ontologies have long been employed in the life sciences to formally represent and reason over domain knowledge and they are employed in almost every major biological database. Recently, ontologies are increasingly being used to provide background knowledge in similarity-based analysis and machine learning models. The methods employed to combine ontologies and machine learning are still novel and actively being developed. We provide an overview over the methods that use ontologies to compute similarity and incorporate them in machine learning methods; in particular, we outline how semantic similarity measures and ontology embeddings can exploit the background knowledge in ontologies and how ontologies can provide constraints that improve machine learning models. The methods and experiments we describe are available as a set of executable notebooks, and we also provide a set of slides and additional resources at https://github.com/bio-ontology-research-group/machine-learning-with-ontologies.},
    issn = {1477-4054},
    doi = {10.1093/bib/bbaa199},
    url = {https://doi.org/10.1093/bib/bbaa199},
    note = {bbaa199},
    eprint = {https://academic.oup.com/bib/advance-article-pdf/doi/10.1093/bib/bbaa199/33875255/bbaa199.pdf},
}
```
