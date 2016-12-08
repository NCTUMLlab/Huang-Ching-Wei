Here contains the testing code for experiment.
We also makes some tool to visualize trained features and results to help evaulating the models.(named by XXFeaturePlot)

In each testing code, we first load the data, set the model coefficient and the structure, call the training function of model and get the return trained parameters and compiled theano function for features and testing.
Finally use the return compiled function to plot the latent feature to see the performance of distribution matching.

Note, you needs to add the path of the Model folder in test file and XXFeaturePlot first.