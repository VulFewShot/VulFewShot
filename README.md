### This is the online repository of 《VulFewShot: Improving Few-Shot Vulnerability Classification by Contrastive Learning》

#### Dataset

The datasets we use are MVD and MVD-part, which are in the compressed file named by the first issue name.

#### Experiment

The core file is the main.py, where the core functions are main() and main_model().

The experiment consisted of the following steps:

1. Static analysis and processing to generate pkl files.

2. Split training and test sets, 10-fold crossover, etc.

3. Run main.py to run and test.

#### Dependency

python version: Python3.8.10
regex==2023.5.5
numpy==1.24.3
torch==2.0.1
scikit-learn==1.1.2
transformers==4.21.3
tqdm==4.65.0
gensim==4.2.0
pickleshare==0.7.5
pandas==2.0.2
jupyter==1.0.0
