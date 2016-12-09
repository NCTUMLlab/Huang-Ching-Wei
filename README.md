# Huang-Ching-Wei
This project contains source code for Dataset, Model, Testing and some utilization used in Deep Variational Transfer Learning and Classification(DVTLC).

We use Python2 and run on Ubunto, we also use the following library: Theano, matplotlib, scikit-learn(optional).
Please make sure you already install these library before runing the project.

We first breiefly describe the content of each folder. For detail, please see the Readme.txt in each folder.

## Folder-Dataset
Here contains code for data genearation and data preprocessing.
We use three data set: Inter-Twining Moon(Synthetic Data), Office(Image), Amazon Reviews(Seniment).
In this project we don't contatins the data of Office and Amazton Reviews.
Please download from the provider website.

Office : https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code (please download the Office-Caltech)

Amazon Reviews : https://www.cs.jhu.edu/~mdredze/datasets/sentiment/ (please download the processed_acl.tar.gz)

## Folder-Model
Here contains code for model construction.
Except the proposed model, we also bulid the following model for comparison:

Deep Nerual Network

Deep Domain Confusion Maximizing for Domain Invariance  https://arxiv.org/abs/1412.3474
    
[MLSP 2015]Deep semi-supervised learning for domain adaptation  http://ieeexplore.ieee.org/document/7324325/?arnumber=7324325&tag=1
    
[JMLR 2016]Domain-Adversarial Training of Neural Networks   http://jmlr.csail.mit.edu/papers/volume17/15-239/15-239.pdf
    
[ICLR 2016]The Variational Fair Autoencoder https://arxiv.org/abs/1511.00830
    
For the proposed model, we have Variational Adversarial Neural Network and Variational Learning for Domain Features.

## Folder-Experiment
Here contains the testing code for experiment.
One can just consider the codes as a example that how to use the model and training.
We also makes some tool to visualize trained features and results to help evaulating the models.

!! Before run the code, please change the path of dataset file and the path of Folder Model in the following codes: every testing codes, XXFeaturePlot.py and DataPackage.py in Amazon_Reviews. And create Folder named "Experimental_Result" at each folders of experiments
