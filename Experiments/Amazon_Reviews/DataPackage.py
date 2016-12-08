from __future__ import print_function

import scipy.io as sio
import numpy as np

#random the data and split the training data into training data and validation data.

def traindata_process(train_fts, train_label, validate_rate):
    train_fts = train_fts.toarray()        
    n = train_fts.shape[0]           
    v_n = int(n * validate_rate)    
    
    #make the validation data balance(data number of any label are same)
    ind = np.where(train_label[:,0] == 1)[0][0:v_n/2]
    validation_fts = train_fts[ind , :]
    validation_label = train_label[ind , :]
    train_fts = np.delete(train_fts, (ind), axis = 0)
    train_label = np.delete(train_label, (ind), axis = 0)    
    
    ind = np.where(train_label[:,1] == 1)[0][0:v_n/2]
    validation_fts = np.append(validation_fts, train_fts[ind , :], axis=0 )
    validation_label = np.append(validation_label, train_label[ind , :], axis=0 )
    train_fts = np.delete(train_fts, (ind), axis = 0)
    train_label = np.delete(train_label, (ind), axis = 0)     
    
    s_n = train_fts.shape[0] 
    index = np.random.permutation(s_n)
    train_fts = train_fts[index, :]
    train_label = train_label[index, :] 
    
    return train_fts, train_label, validation_fts, validation_label


def datapackage(source, target, max_feature, tfidf_setting, validate_rate=0.2):
    np.random.seed(123)
    
    #please change the file path to your own path
    if tfidf_setting == 'union':
        source_file = '/home/cwhuang/DVTL/Dataset/Amazon_Reviews/Set_union_mf'+str(max_feature)+'/'+source+'_mf'+str(max_feature)+'.npy'                
        data = np.load(source_file)
        train_fts, train_label = data[0]
        test_fts, test_label = data[1]
        test_fts = test_fts.toarray()
        train_fts, train_label, validation_fts, validation_label = traindata_process(train_fts, train_label, validate_rate)
        source_data=[(train_fts, train_label), (validation_fts, validation_label), (test_fts, test_label)]
                
        target_file = '/home/cwhuang/DVTL/Dataset/Amazon_Reviews/Set_union_mf'+str(max_feature)+'/'+target+'_mf'+str(max_feature)+'.npy'
        data = np.load(target_file)
        train_fts, train_label = data[0]
        test_fts, test_label = data[1]     
        test_fts = test_fts.toarray()
        train_fts, train_label, validation_fts, validation_label = traindata_process(train_fts, train_label, validate_rate)
        target_data=[(train_fts, train_label), (validation_fts, validation_label), (test_fts, test_label)]        
        
        
    elif tfidf_setting == 'seperate':
        file = '/home/cwhuang/DVTL/Dataset/Amazon_Reviews/Set_seperate_mf'+str(max_feature)+'/'+source+'2'+target+'_mf'+str(max_feature)+'.npy'                
        data = np.load(file)
        #source
        train_fts, train_label = data[0]                
        test_fts, test_label = data[2]
        test_fts = test_fts.toarray()
        train_fts, train_label, validation_fts, validation_label = traindata_process(train_fts, train_label, validate_rate)
        source_data=[(train_fts, train_label), (validation_fts, validation_label), (test_fts, test_label)]
        
        #target
        train_fts, train_label = data[1]                
        test_fts, test_label = data[3]
        test_fts = test_fts.toarray()
        train_fts, train_label, validation_fts, validation_label = traindata_process(train_fts, train_label, validate_rate)
        target_data=[(train_fts, train_label), (validation_fts, validation_label), (test_fts, test_label)]     
        
    return source_data, target_data
