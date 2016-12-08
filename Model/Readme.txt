In this folder, the codes which name is lower case contain basic component and utility functionwhich can be used to construct model; the codes which name is capital contain complete models.

Basic component:
[a] criteria.py
Here contains the symbolic functions of different criteria.

[b] nnet.py
Here contains basic components which is used to construct neural networks.
We have class HiddenLayer, which represents the typical hidden layer of a NN: units are fully-connected and have activation function.
We can use several HiddenLayer to construct a deep fully connected nerural network, which is the class NN_Block.
          
For variational auto-encoder(VAE), we create sample layer of Gaussian and categorical distribution which presents the sampling process of Stochastic Gradient Variational Bayes(SVGB).
In VAE, we use nerual network to simulate the probabilistic encoder/decoder, which is the class Gauusian_MLP.
          
We also have the adam update function.
The code of adam update is adapted from https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py.

Utility function:
[a] util.py
some utility function

[b] tsne.py
The code is from https://lvdmaaten.github.io/tsne/.
We use t-sne to run the dimension reduction for visualizing feature.

Complete Model:
We have create complete models of NN, DANN, DSSL, VFAE, VANN, VLDF. (VLDF_ANN uses the ANN to replace the MMD)
For each model, we using the basic component to create the symbolic function of the models, criteria and update function.
We also create function: XX_training (XX refers to the model name), to train the model by giving the input and coefficient.
It will returns the trained parameters and the complied theano function for features and testing

Note that in the experiment, because the numbers of training data, validation data and test data are different, the according symbolic and complied theano functions are also different.
We create different symbolic and complied theano functions for each data. 
To make the parameters be the same among these functions, we create the structure for parameters (named XX_params) and use it when creating the symbolic function.
