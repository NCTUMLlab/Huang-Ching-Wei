from __future__ import print_function

import matplotlib.pyplot as plt

import os
import sys
import timeit

import scipy.io as sio
import scipy.stats as spy
import numpy as np
import theano
import theano.tensor as T
import tsne
################################################################################################################
################################################################################################################

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y        

def feature_tsne(f_S, f_T):
    N_S = np.shape(f_S)[0]
    N_T = np.shape(f_T)[0]
    N = N_S + N_T
    f_dim = np.shape(f_S)[1]
    f = np.append(f_S, f_T, axis = 0)
    
    fr = tsne.tsne(
        X = f,
        no_dims = 2,
        initial_dims = f_dim,
        perplexity = 30.0
    )
    
    fr_S = fr[0:N_S, :]
    fr_T = fr[N_S:N_S+N_T, :]
    
    return [fr_S, fr_T]

def data2plot(title, fts, label, color, marker, line=False, legend=False, plot_enable=True):
    """
    given the data, create plot.
    """
    #chart create
    color_chart = ['red', 'blue', 'green', 'c', 'm', 'y', 'k', '#00ff77', '#ff0077', '#770055']
    marker_chart = ['None', 'o', 'x', '*']
    if line:
        linestyle='-'
    else:
        linestyle='None'
        
    #Python plot 
    markersize = 20
    fig, ax = plt.subplots(figsize=(30, 30))
    for i in range(len(fts)/2):
        ax.plot(fts[i*2], fts[i*2+1], color=color_chart[color[i]-1], marker=marker_chart[marker[i]],
            linestyle=linestyle, label=label[i], markersize=markersize)  
    if legend:
        plt.legend(fontsize='xx-large')
    filename = './Experimental_Result/%s.png' % (title)
    plt.savefig(filename)    
    
    if plot_enable == False:
        plt.close(fig)
    
                       
                
                
                

                
        
    
    