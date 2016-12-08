from __future__ import print_function

import os
import sys
import timeit
import pickle

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

sys.path.insert(0, "/home/cwhuang/DVTL/Model/")
import nnet as nn
import criteria	as er
import util

def features_plot(features_model, test_model, source_data, target_data, sample_n, description, plot_enable=True):
                
    train_fts_source, train_labels_source = source_data[0]
    valid_fts_source, valid_labels_source = source_data[1]
    test_fts_source, test_labels_source = source_data[2]
    
    train_fts_target, train_labels_target = target_data[0]
    valid_fts_target, valid_labels_target = target_data[1]
    test_fts_target, test_labels_target = target_data[2]        
    
    y_dim = np.shape(train_labels_source)[1]
    S_labels = train_labels_source[0:sample_n, :]
    T_labels = train_labels_target[0:sample_n, :]

    zy_S = features_model()[0][0:sample_n, :]
    zy_T = features_model()[1][0:sample_n, :]
    
    zy_S, zy_T = util.feature_tsne(zy_S, zy_T)
    
    label_zy_S = []
    label_zy_T = []

    for i in range(y_dim):
        label_zy_S.append( zy_S[np.where(S_labels[:,i] == 1)[0], :] )
        label_zy_T.append( zy_T[np.where(T_labels[:,i] == 1)[0], :] )

    #Source zy feature
    title = 'Source_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts+(label_zy_S[i][:,0], label_zy_S[i][:,1])
    label = ['negative', 'positive']
    color = [1, 2]
    marker = [1, 1]
    line = False   
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=plot_enable)            
        
    #Target zy feature
    title = 'Target_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts+(label_zy_T[i][:,0], label_zy_T[i][:,1])
    label = ['negative', 'positive']
    color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    marker = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    line = False 
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=plot_enable) 
    
    #Both source, target zy feature
    title = 'Zy_feature_%s' % (description)
    fts = ()
    tmp = ()
    for i in range(y_dim):
        fts = fts+(label_zy_S[i][:,0], label_zy_S[i][:,1])
        fts = fts+(label_zy_T[i][:,0], label_zy_T[i][:,1])
    label = ['Source:negative', 'Target:negative', 'Source:positive', 'Target:positive']
    color = [1, 1, 2, 2]
    marker = [1, 2, 1, 2]
    line = False 
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend, plot_enable=plot_enable)  
