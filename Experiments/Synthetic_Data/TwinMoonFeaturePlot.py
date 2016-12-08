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
import util

def features_plot(features_model, test_model, source_data, target_data, sample_n, description, reconstruction=False):

    train_fts_source, train_labels_source = source_data[0]
    valid_fts_source, valid_labels_source = source_data[1]
    test_fts_source, test_labels_source = source_data[2]
    
    train_fts_target, train_labels_target = target_data[0]
    valid_fts_target, valid_labels_target = target_data[1]
    test_fts_target, test_labels_target = target_data[2]        
    
    y_dim = np.shape(train_labels_source)[1]
    S_labels = train_labels_source
    T_labels = train_labels_target         

    #-------------------------------------------------------------------------------------------        
    
    zy_S = features_model()[0][0:sample_n,:]
    zy_T = features_model()[1][0:sample_n,:]
    
    zy_S, zy_T = util.feature_tsne(zy_S, zy_T)
    
    label_zy_S = []
    label_zy_T = []        
    for i in range(y_dim):
        label_zy_S.append( zy_S[np.where(S_labels[0:sample_n,i] == 1)[0], :] )
        label_zy_T.append( zy_T[np.where(T_labels[0:sample_n,i] == 1)[0], :] )
            
    #Source zy feature
    title = 'Source_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts+(label_zy_S[i][:,0], label_zy_S[i][:,1])
    label = ('label 0', 'label 1')
    color = [1, 2]
    marker = [1, 1]
    line = False   
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line)
       
    #Target zy feature
    title = 'Target_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts+(label_zy_T[i][:,0], label_zy_T[i][:,1])
    label = ('label 0', 'label 1')
    color = [1, 2]
    marker = [2, 2]
    line = False   
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line)    
      
    #Both source, target zy feature
    title = 'Zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts+(label_zy_S[i][:,0], label_zy_S[i][:,1])
        fts = fts+(label_zy_T[i][:,0], label_zy_T[i][:,1])
    label = ('Source label 0', 'Target label 0', 'Source label 1', 'Target label 1')
    color = [1, 1, 2, 2]
    marker = [1, 2, 1, 2]
    line = False
    legend = False
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend)    
             
    #print('-------------------------------------------------------------------------') 
    #Classification Result of Training Data        
    S_labels_predict = features_model()[2]
    T_labels_predict = features_model()[3]
             
    S_0 = train_fts_source[np.where(S_labels_predict == 0)[0], :]
    S_1 = train_fts_source[np.where(S_labels_predict == 1)[0], :]
    T_0 = train_fts_target[np.where(T_labels_predict == 0)[0], :]
    T_1 = train_fts_target[np.where(T_labels_predict == 1)[0], :]

    title = 'Train_classification_%s' % (description)
    fts = (S_0[:,0], S_0[:,1], S_1[:,0], S_1[:,1], T_0[:,0], T_0[:,1], T_1[:,0], T_1[:,1])
    label = ('Source label 0', 'Target label 0', 'Source label 1', 'Target label 1')
    color = [1, 2, 1, 2]
    marker = [1, 1, 2, 2]
    line = False   
    legend = False
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend)                        
             
    #Classification Result of Testing Data             
    S_labels_predict = test_model()[3]
    T_labels_predict = test_model()[4]
                 
    S_0 = test_fts_source[np.where(S_labels_predict == 0)[0], :]
    S_1 = test_fts_source[np.where(S_labels_predict == 1)[0], :]
    T_0 = test_fts_target[np.where(T_labels_predict == 0)[0], :]
    T_1 = test_fts_target[np.where(T_labels_predict == 1)[0], :]
             
    title = 'Test_classification_%s' % (description)
    fts = (S_0[:,0], S_0[:,1], S_1[:,0], S_1[:,1], T_0[:,0], T_0[:,1], T_1[:,0], T_1[:,1])
    label = ('Source label 0', 'Target label 0', 'Source label 1', 'Target label 1')
    color = [1, 2, 1, 2]
    marker = [1, 1, 2, 2]
    line = False   
    legend = False
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend = legend)      

    print('-------------------------------------------------------------------------') 
    
    #Reconstruction of generative model(VFAE, VLDF)
    if reconstruction == False:
        return
    
    S_recon_x = features_model()[4]
    T_recon_x = features_model()[5]
   
    title = 'Reconstruction_%s' % (description)
    fts = (S_recon_x[:,0],  S_recon_x[:,1], T_recon_x[:,0],  T_recon_x[:,1])    
    label = ('Source', 'Target')
    color = [2, 2]
    marker = [1, 2]
    line = False    
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend=legend)       
