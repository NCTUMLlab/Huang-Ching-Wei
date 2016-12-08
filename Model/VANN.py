from __future__ import print_function

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

import nnet as nn
import criteria as er
import util

################################################################################################################
################################################################################################################

class VANN_struct(object):
    def __init__(self):
        self.encoder1 = nn.GMLP_struct()
        self.encoder2 = nn.GMLP_struct()
        self.encoder3 = nn.NN_struct()
        self.decoder1 = nn.GMLP_struct()
        self.decoder2 = nn.GMLP_struct()       
        self.DomainClassifier = nn.NN_struct()        

class VANN_coef(object):
    def __init__(self, alpha = 100, beta = 100, L = 1, optimize = 'Adam_update'):
        self.alpha = alpha                     # weight for classification loss
        self.beta = beta                       # weight for MMD 
        self.L = L                             # sample number        
        self.optimize = optimize               # option of optimization     

class VANN_params(object):
    def __init__(self):
        self.EC1_params=None
        self.EC2_params=None
        self.EC3_params=None
        self.DC1_params=None
        self.DC2_params=None
        self.DoC_params=None
        
    def update_value(self, params_name, params_value, struct):
        
        params = [ theano.shared(value=value, name=name, borrow=True)
            for value, name in zip(params_value, params_name)]
        
        self.EC1_params = []
        self.EC2_params = []
        self.EC3_params = []
        self.DC1_params = []
        self.DC2_params = []
        self.DoC_params = []
        
        i = 0;        
        params_num = [len(struct.encoder1.share.activation), len(struct.encoder1.mu.activation), len(struct.encoder1.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.EC1_params.append(tmp)
            
        params_num = [len(struct.encoder2.share.activation), len(struct.encoder2.mu.activation), len(struct.encoder2.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.EC2_params.append(tmp)      
        
        for j in range(len(struct.encoder3.activation)):
            self.EC3_params.append(params[i])
            i = i+1        
            self.EC3_params.append(params[i])
            i = i+1        
        
        
        params_num = [len(struct.decoder1.share.activation), len(struct.decoder1.mu.activation), len(struct.decoder1.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.DC1_params.append(tmp)
            
        params_num = [len(struct.decoder2.share.activation), len(struct.decoder2.mu.activation), len(struct.decoder2.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.DC2_params.append(tmp)    

        for j in range(len(struct.DomainClassifier.activation)):
            self.DoC_params.append(params[i])
            i = i+1        
            self.DoC_params.append(params[i])
            i = i+1            

    def update_symbol(self, params, struct):
        
        self.EC1_params = []
        self.EC2_params = []
        self.EC3_params = []
        self.DC1_params = []
        self.DC2_params = []
        self.DoC_params = []
        
        i = 0;        
        params_num = [len(struct.encoder1.share.activation), len(struct.encoder1.mu.activation), len(struct.encoder1.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.EC1_params.append(tmp)
            
        params_num = [len(struct.encoder2.share.activation), len(struct.encoder2.mu.activation), len(struct.encoder2.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.EC2_params.append(tmp)      
        
        for j in range(len(struct.encoder3.activation)):
            self.EC3_params.append(params[i])
            i = i+1        
            self.EC3_params.append(params[i])
            i = i+1        
        
        
        params_num = [len(struct.decoder1.share.activation), len(struct.decoder1.mu.activation), len(struct.decoder1.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.DC1_params.append(tmp)
            
        params_num = [len(struct.decoder2.share.activation), len(struct.decoder2.mu.activation), len(struct.decoder2.sigma.activation)]
        for k in range(len(params_num)):
            tmp=[]
            for j in range(params_num[k]):
                tmp.append(params[i])
                i = i+1
                tmp.append(params[i])
                i = i+1
            self.DC2_params.append(tmp)    
            
        for j in range(len(struct.DomainClassifier.activation)):
            self.DoC_params.append(params[i])
            i = i+1        
            self.DoC_params.append(params[i])
            i = i+1                  
        
'''Model Definition/Construct'''

class VANN(object):   
    """
    The semi-supervised model Domain-Adversial Variational Autoencoder
    To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
    The domain label s will constuct inside this class
    For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
    Encoder refer to Encoder NN, Decoder refer to Decoder NN    
    """

    def __init__(self, rng, input_source, input_target, label_source, batch_size, struct, coef, train = False, init_params=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input_source: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
        
        :type input_target: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)        

        :type xxx_struct: class NN_struct
        :param xxx_strucat: define the structure of each NN
        """
        
        if train == True:
            batch_size[0] = batch_size[0] * coef.L
            batch_size[1] = batch_size[1] * coef.L
            tmp_S = input_source
            tmp_T = input_target
            tmp_l = label_source
            for i in range(coef.L-1):
                tmp_S = T.concatenate( [tmp_S, input_source], axis = 0)
                tmp_T = T.concatenate( [tmp_T, input_target], axis = 0)
                tmp_l = T.concatenate( [tmp_l, label_source], axis = 0)
            input_source = tmp_S
            input_target = tmp_T
            label_source = tmp_l
            L = coef.L
        else:
            L = 1
        
        self.L = L        
        
        self.struct = struct
        encoder1_struct = struct.encoder1
        encoder2_struct = struct.encoder2
        encoder3_struct = struct.encoder3
        decoder1_struct = struct.decoder1
        decoder2_struct = struct.decoder2
        DoC_struct = struct.DomainClassifier
        
        alpha = coef.alpha
        beta = coef.beta
        optimize = coef.optimize       
        
        if init_params == None:
            init_params = VANN_params()            
        
        #------------------------------------------------------------------------
        #Encoder 1 Neural Network: present q_\phi({z_y}_n | x_n, d_n)
        zero_v_S = T.zeros([batch_size[0],1], dtype=theano.config.floatX)
        zero_v_T = T.zeros([batch_size[1],1], dtype=theano.config.floatX)
        one_v_S = T.ones([batch_size[0],1], dtype=theano.config.floatX)
        one_v_T = T.ones([batch_size[1],1], dtype=theano.config.floatX)
        
        d_source = T.concatenate([zero_v_S, one_v_S], axis=1)
        xd_source = T.concatenate([input_source, d_source], axis=1)
        d_target = T.concatenate([one_v_T, zero_v_T], axis=1)
        xd_target = T.concatenate([input_target, d_target], axis=1)
                    
        
        self.Encoder1 = nn.Gaussian_MLP(
            rng=rng,
            input_source=xd_source,
            input_target=xd_target,
            struct = encoder1_struct,
            batch_size = batch_size,
            params = init_params.EC1_params,
            name='Encoder1'
        )
        
        zy_dim = encoder1_struct.mu.layer_dim[-1]
        self.EC_zy_S_mu = self.Encoder1.S_mu
        self.EC_zy_S_log_sigma = self.Encoder1.S_log_sigma
        self.EC_zy_S_sigma = T.exp(self.EC_zy_S_log_sigma)
        self.EC_zy_T_mu = self.Encoder1.T_mu
        self.EC_zy_T_log_sigma = self.Encoder1.T_log_sigma
        self.EC_zy_T_sigma = T.exp(self.EC_zy_T_log_sigma)
        
        self.zy_S = self.Encoder1.S_output
        self.zy_T = self.Encoder1.T_output
        
        self.Encoder1_params = self.Encoder1.params
        self.Encoder1_learning_rate = self.Encoder1.learning_rate     
        self.Encoder1_decay = self.Encoder1.decay      
        
        #------------------------------------------------------------------------
        #Encoder 3 Neural Network: present q_\phi(y_n | {z_1}_n)
        self.Encoder3_pi = nn.NN_Block(
            rng=rng,            
            input_source=self.zy_S,
            input_target=self.zy_T,
            struct = encoder3_struct,
            params = init_params.EC3_params,
            name='Encoder3_pi'
        )        
        
        #Sample layer
        self.EC_3_CSL_target = nn.CatSampleLayer(
            pi=self.Encoder3_pi.output_target,
            n_in = encoder3_struct.layer_dim[-1],
            batch_size = batch_size[1] 
        )
        y_dim = encoder3_struct.layer_dim[-1]
        self.EC_y_S_pi = self.Encoder3_pi.output_source
        self.EC_y_T_pi = self.Encoder3_pi.output_target
        
        self.y_T = self.EC_3_CSL_target.output
        
        self.Encoder3_params = self.Encoder3_pi.params        
        self.Encoder3_learning_rate = self.Encoder3_pi.learning_rate     
        self.Encoder3_decay = self.Encoder3_pi.decay     
        
        #------------------------------------------------------------------------
        #Encoder 2 Neural Network: present q_\phi({a_y}_n | {z_y}_n, y_n)    
        #Input Append        
        zyy_source = T.concatenate([self.zy_S, label_source], axis=1)        
        zyy_target = T.concatenate([self.zy_T, self.y_T], axis=1)   
        
        self.Encoder2 = nn.Gaussian_MLP(
            rng=rng,
            input_source=zyy_source,
            input_target=zyy_target,
            struct = encoder2_struct,
            batch_size = batch_size,
            params = init_params.EC2_params,
            name='Encoder2'
        )        
        
        ay_dim = encoder2_struct.mu.layer_dim[-1]
        self.EC_ay_S_mu = self.Encoder2.S_mu
        self.EC_ay_S_log_sigma = self.Encoder2.S_log_sigma
        self.EC_ay_S_sigma = T.exp(self.EC_ay_S_log_sigma)
        self.EC_ay_T_mu = self.Encoder2.T_mu
        self.EC_ay_T_log_sigma = self.Encoder2.T_log_sigma
        self.EC_ay_T_sigma = T.exp(self.EC_ay_T_log_sigma)
        
        self.ay_S = self.Encoder2.S_output
        self.ay_T = self.Encoder2.T_output
       
        self.Encoder2_params = self.Encoder2.params
        self.Encoder2_learning_rate = self.Encoder2.learning_rate     
        self.Encoder2_decay = self.Encoder2.decay        
        
        #------------------------------------------------------------------------
        #Decoder 1 Neural Network: present p_\theta(x_n | {z_1}_n, s_n)
        zyd_source = T.concatenate([self.zy_S, d_source], axis=1)
        zyd_target = T.concatenate([self.zy_T, d_target], axis=1)         
                
        self.Decoder1 = nn.Gaussian_MLP(
            rng=rng,
            input_source=zyd_source,
            input_target=zyd_target,
            struct = decoder1_struct,
            batch_size = batch_size,
            params = init_params.DC1_params,
            name='Decoder1'
        )   
        
        x_dim = decoder1_struct.mu.layer_dim[-1]
        self.DC_x_S_mu = self.Decoder1.S_mu
        self.DC_x_S_log_sigma = self.Decoder1.S_log_sigma
        self.DC_x_S_sigma = T.exp(self.DC_x_S_log_sigma)
        self.DC_x_T_mu = self.Decoder1.T_mu
        self.DC_x_T_log_sigma = self.Decoder1.T_log_sigma
        self.DC_x_T_sigma = T.exp(self.DC_x_T_log_sigma)
                
        self.Decoder1_params = self.Decoder1.params
        self.Decoder1_learning_rate = self.Decoder1.learning_rate     
        self.Decoder1_decay = self.Decoder1.decay      
        
        #------------------------------------------------------------------------
        #Decoder 2 Neural Network: present p_\theta({z_y}_n | {a_y}_n, y_n)
        ayy_source = T.concatenate([self.ay_S, label_source], axis=1)        
        ayy_target = T.concatenate([self.ay_T, self.y_T], axis=1)         

        self.Decoder2 = nn.Gaussian_MLP(
            rng=rng,
            input_source=ayy_source,
            input_target=ayy_target,
            struct = decoder2_struct,
            batch_size = batch_size,
            params = init_params.DC2_params,
            name='Decoder2'
        )         

        self.DC_zy_S_mu = self.Decoder2.S_mu
        self.DC_zy_S_log_sigma = self.Decoder2.S_log_sigma
        self.DC_zy_S_sigma = T.exp(self.DC_zy_S_log_sigma)
        self.DC_zy_T_mu = self.Decoder2.T_mu
        self.DC_zy_T_log_sigma = self.Decoder2.T_log_sigma
        self.DC_zy_T_sigma = T.exp(self.DC_zy_T_log_sigma)
        
        self.Decoder2_params = self.Decoder2.params
        self.Decoder2_learning_rate = self.Decoder2.learning_rate     
        self.Decoder2_decay = self.Decoder2.decay 
        
        #------------------------------------------------------------------------
        #Domain Clasiifier 3 Neural Network: present p_\varphi(d=0|z_y)
        self.DomainClassifier = nn.NN_Block(
            rng=rng,            
            input_source=self.zy_S,
            input_target=self.zy_T,
            struct = DoC_struct,
            params = init_params.DoC_params,
            name='DomainClassifier'
        )
        
        self.DoC_output_S = self.DomainClassifier.output_source
        self.DoC_output_T = self.DomainClassifier.output_target
        
        self.DoC_params = self.DomainClassifier.params
        self.DoC_learning_rate = self.DomainClassifier.learning_rate     
        self.DoC_decay = self.DomainClassifier.decay         
        
        #------------------------------------------------------------------------
        # Error Function Set                
        # KL(q(zy)||p(zy)) -----------
        self.KL_zy_source = er.KLGaussianGaussian(self.EC_zy_S_mu, self.EC_zy_S_log_sigma, self.DC_zy_S_mu, self.DC_zy_S_log_sigma).sum() 
        self.KL_zy_target = er.KLGaussianGaussian(self.EC_zy_T_mu, self.EC_zy_T_log_sigma, self.DC_zy_T_mu, self.DC_zy_T_log_sigma).sum()         
        
        # KL(q(ay)||p(ay)) -----------     
        self.KL_ay_source = er.KLGaussianStdGaussian(self.EC_ay_S_mu, self.EC_ay_S_log_sigma).sum()
        self.KL_ay_target = er.KLGaussianStdGaussian(self.EC_ay_T_mu, self.EC_ay_T_log_sigma).sum() 
                
        # KL(q(y)||p(y)) only target data-----------
        # prior of y is set to 1/K, K is category number
        threshold = 0.0000001        
        pi_0 = T.ones([batch_size[1], y_dim], dtype=theano.config.floatX) / y_dim
        self.KL_y_target = T.sum(- self.EC_y_T_pi * T.log( T.maximum(self.EC_y_T_pi / pi_0, threshold)), axis=1).sum() 
                 
        # classification error only source data-----------
        self.LH_y_source = - T.sum(- label_source * T.log( T.maximum(self.EC_y_S_pi, threshold)), axis=1).sum() 
        
        # Likelihood p(x) ----------- if gaussian
        self.LH_x_source = er.LogGaussianPDF(input_source, self.DC_x_S_mu, self.DC_x_S_log_sigma).sum()
        self.LH_x_target = er.LogGaussianPDF(input_target, self.DC_x_T_mu, self.DC_x_T_log_sigma).sum()
        
        # Domain classification error  smaller, better
        self.DoC_error_S = T.sum(- d_source * T.log( T.maximum(self.DoC_output_S, threshold)), axis=1).sum()
        self.DoC_error_T = T.sum(- d_target * T.log( T.maximum(self.DoC_output_T, threshold)), axis=1).sum()
        self.DoC_error = self.DoC_error_S + self.DoC_error_T
                
        #Cost function
        '''
        tmp = self.KL_zy_source + self.KL_zy_target + self.KL_ay_source + self.KL_ay_target \
            + self.LH_x_source + self.LH_x_target + self.KL_y_target + self.LH_y_source * alpha +  self.DoC_error * beta                
        self.cost = -tmp.mean()
        '''
        
        LM_tmp = self.KL_zy_source + self.KL_zy_target + self.KL_ay_source + self.KL_ay_target \
            + self.LH_x_source + self.LH_x_target + self.KL_y_target + self.LH_y_source * alpha
        self.LM_cost = -LM_tmp / (batch_size[0]+batch_size[1])
        
        DoC_tmp = self.DoC_error
        self.DoC_cost = -DoC_tmp.mean() / (batch_size[0]+batch_size[1]) * beta
        
        self.cost = self.LM_cost + self.DoC_cost
        
        # the parameters of the latent model
        self.LM_params = self.Encoder1_params + self.Encoder2_params + self.Encoder3_params + self.Decoder1_params + self.Decoder2_params
        self.params = self.LM_params + self.DoC_params
        
        self.LM_learning_rate = self.Encoder1_learning_rate + self.Encoder2_learning_rate + self.Encoder3_learning_rate \
        + self.Decoder1_learning_rate + self.Decoder2_learning_rate
        self.learning_rate = self.LM_learning_rate + self.DoC_learning_rate      
        
        self.LM_decay = self.Encoder1_decay + self.Encoder2_decay + self.Encoder3_decay + self.Decoder1_decay + self.Decoder2_decay
        self.decay = self.LM_decay + self.DoC_decay        
        
        if optimize == 'Adam_update':
            #Adam update function
            self.LM_updates = nn.adam(
                loss = self.cost,
                all_params = self.LM_params,
                all_learning_rate= self.LM_learning_rate
            )        
            
            self.DoC_updates = nn.adam(
                loss = -self.DoC_cost,                   #Adversarial process
                all_params = self.DoC_params,
                all_learning_rate= self.DoC_learning_rate
            )
            
        elif optimize == 'SGD':
            #Standard update function
            LM_gparams = [T.grad(self.cost, param) for param in self.LM_params]
        
            self.LM_params_updates = [
                (LM_param, LM_param - learning_rate * LM_gparam)
                for LM_param, LM_gparam, learning_rate in zip(self.LM_params, LM_gparams, self.LM_learning_rate)
            ]       

            self.LM_learning_rate_update = [
                (learning_rate, learning_rate * decay)
                for learning_rate, decay in zip(self.LM_learning_rate, self.LM_decay)
            ]            
            
            self.LM_updates = self.LM_params_updates + self.LM_learning_rate_update
            
            DoC_gparams = [T.grad(-self.DoC_cost, param) for param in self.DoC_params]
        
            self.DoC_params_updates = [
                (DoC_param, DoC_param - learning_rate * DoC_gparam)
                for DoC_param, DoC_gparam, learning_rate in zip(self.DoC_params, DoC_gparams, self.DoC_learning_rate)
            ]             
            
            self.DoC_learning_rate_update = [
                (learning_rate, learning_rate * decay)
                for learning_rate, decay in zip(self.DoC_learning_rate, self.DoC_decay)
            ]
            
            self.DoC_updates = self.DoC_params_updates + self.DoC_learning_rate_update


        # keep track of model input
        self.input_source = input_source
        self.input_target = input_target            
        
        #Predict Label
        self.y_pred_source = T.argmax(self.EC_y_S_pi, axis=1)
        self.y_pred_target = T.argmax(self.EC_y_T_pi, axis=1)

    def source_reconstruct(self):
        return self.DC_x_S_mu
    
    def target_reconstruct(self):
        return self.DC_x_T_mu        
        
    def feature_outputs(self):
        return [self.EC_zy_S_mu, self.EC_zy_T_mu]          

    def source_predict_raw(self):
        return self.EC_y_S_pi
    
    def target_predict_raw(self):
        return self.EC_y_T_pi       
        
    def source_predict(self):
        return self.y_pred_source
    
    def target_predict(self):
        return self.y_pred_target 
        
    def source_errors(self, y):
        #Classification Error
        tmp = y
        for i in range(self.L-1):
            tmp = T.concatenate( [tmp, y], axis = 0)    
        y = tmp        
        return T.mean(T.neq(self.y_pred_source, T.argmax(y, axis=1)))

    def target_errors(self, y):
        #Classification Error
        tmp = y
        for i in range(self.L-1):
            tmp = T.concatenate( [tmp, y], axis = 0)    
        y = tmp        
        return T.mean(T.neq(self.y_pred_target, T.argmax(y, axis=1)))
    
    def params_name(self):
        params_name = ( param.name for param in self.params)
        
        return params_name
    
    def params_value(self):
        params_value = ( param.get_value() for param in self.params)
        
        return params_value
    
    def params_symbol(self):
        tmp = VANN_params()
        tmp.update_symbol(self.params, self.struct)
        return tmp     
    
def VANN_training(source_data, target_data, n_train_batches, n_epochs, k, struct, coef, description, process_display=True):
             
    #########################################################
    ###                        Data                       ###
    #########################################################               
                        
    train_ftd_source, train_labeld_source = source_data[0]
    valid_ftd_source, valid_labeld_source = source_data[1]
    test_ftd_source, test_labeld_source = source_data[2]
    
    train_ftd_target, train_labeld_target = target_data[0]
    valid_ftd_target, valid_labeld_target = target_data[1]
    test_ftd_target, test_labeld_target = target_data[2] 
           
    train_ftd_source, train_labeld_source = util.shared_dataset((train_ftd_source, train_labeld_source))
    valid_ftd_source, valid_labeld_source = util.shared_dataset((valid_ftd_source, valid_labeld_source))
    test_ftd_source, test_labeld_source = util.shared_dataset((test_ftd_source, test_labeld_source))
    
    train_ftd_target, train_labeld_target = util.shared_dataset((train_ftd_target, train_labeld_target))
    valid_ftd_target, valid_labeld_target = util.shared_dataset((valid_ftd_target, valid_labeld_target))
    test_ftd_target, test_labeld_target = util.shared_dataset((test_ftd_target, test_labeld_target))
                 
    batch_size_S = train_ftd_source.get_value(borrow=True).shape[0] // n_train_batches
    batch_size_T = train_ftd_target.get_value(borrow=True).shape[0] // n_train_batches
    validate_S_size = valid_ftd_source.get_value(borrow=True).shape[0]
    validate_T_size = valid_ftd_target.get_value(borrow=True).shape[0]
    test_S_size = test_ftd_source.get_value(borrow=True).shape[0]
    test_T_size = test_ftd_target.get_value(borrow=True).shape[0]
    print(
        'number of minibatch at one epoch: %i, batch size source : %i, target : %i \n validation size, S:%i, T:%i, test size, S:%i, T:%i' %
        (n_train_batches, batch_size_S, batch_size_T, validate_S_size, validate_T_size, test_S_size, test_T_size)
    )   
     
    #######################################################################
    ###                        BUILD ACTUAL MODEL                       ###
    #######################################################################
        
    print('... building the model')
    
    # allocate symbolic variables for the data
    #index_source = T.lscalar()  # index to a [mini]batch
    #index_target = T.lscalar()  # index to a [mini]batch
    index = T.lscalar()  # index to a [mini]batch
    x_source = T.matrix('x_source')  # the data is presented as rasterized images
    y_source = T.matrix('y_source')  # the labels are presented as signal vector 
    x_target = T.matrix('x_target')  # the data is presented as rasterized images
    y_target = T.matrix('y_target')  # the labels are presented as signal vector    
    
    rng = np.random.RandomState(1234)
               
    # construct the VANN class
    classifier = VANN(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [batch_size_S, batch_size_T],
        struct = struct,
        coef = coef,
        train = False 
    )    

    validate_classifier = VANN(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [validate_S_size, validate_T_size],
        struct = struct,
        coef = coef,
        init_params = classifier.params_symbol()
    )    

    test_classifier =  VANN(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [test_S_size, test_T_size],
        struct = struct,
        coef = coef,
        init_params = classifier.params_symbol()
    )    
              
    test_model = theano.function(
        inputs=[],
        outputs=[test_classifier.LM_cost, test_classifier.source_errors(y_source), test_classifier.target_errors(y_target), 
                 test_classifier.source_predict(), test_classifier.target_predict()],
        givens={
            x_source: test_ftd_source,
            y_source: test_labeld_source,
            x_target: test_ftd_target,
            y_target: test_labeld_target
        }        
    )
    
    validate_model = theano.function(
        inputs=[],
        outputs=[validate_classifier.LM_cost, validate_classifier.source_errors(y_source), validate_classifier.target_errors(y_target), 
                 validate_classifier.source_predict_raw(), validate_classifier.target_predict_raw()],
        givens={
            x_source: valid_ftd_source,
            y_source: valid_labeld_source,
            x_target: valid_ftd_target,
            y_target: valid_labeld_target
        }        
    )                
    
    validate_bytraindata_model = theano.function(
        inputs=[index],
        outputs=[classifier.LM_cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict(), classifier.target_predict()],
        givens={
            x_source: train_ftd_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            y_source: train_labeld_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            x_target: train_ftd_target[index * batch_size_T : (index + 1) * batch_size_T, :],
            y_target: train_labeld_target[index * batch_size_T : (index + 1) * batch_size_T, :]             
        }       
    )     

    #update function
    LM_updates = classifier.LM_updates    
    DoC_updates = classifier.DoC_updates        
    
    train_latent_model = theano.function(
        inputs=[index],
        outputs=[classifier.LM_cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        updates=LM_updates,
        givens={
            x_source: train_ftd_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            y_source: train_labeld_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            x_target: train_ftd_target[index * batch_size_T : (index + 1) * batch_size_T, :],
            y_target: train_labeld_target[index * batch_size_T : (index + 1) * batch_size_T, :]          
        }       
    )    

    train_DoC_model = theano.function(
        inputs=[index],
        outputs=[classifier.LM_cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        updates=DoC_updates,
        givens={
            x_source: train_ftd_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            y_source: train_labeld_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            x_target: train_ftd_target[index * batch_size_T : (index + 1) * batch_size_T, :],
            y_target: train_labeld_target[index * batch_size_T : (index + 1) * batch_size_T, :]           
        }       
    )     
    
    ################################################################
    ###                        TRAIN MODEL                       ###
    ################################################################
    '''
    Define :
        xx_loss : Cost function value
        xx_score : Classification accuracy rate        
    '''        
    
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(1, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    validation_frequency = n_train_batches
    
    best_iter = 0
    best_train_loss = np.inf
    best_validation_loss = np.inf  
    test_loss = np.inf
    train_score = 0.
    validation_score = 0.
    test_score = 0.    
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    train_losses_record =[]
    validate_losses_record =[]

    test_losses = test_model()[1]
    test_score_S = 1 - np.mean(test_losses)
    test_losses =test_model()[2]
    test_score_T = 1 - np.mean(test_losses)
                                        
    print(('Initial, test accuracy: source domain :%f %%, target domain %f %%') %
            (test_score_S * 100., test_score_T * 100.))    
    
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):        
            
            # first train domain classifier
            for step in range(k):
                train_DoC_model((minibatch_index+step)%n_train_batches)            
            
            minibatch_avg_cost = train_latent_model(minibatch_index)[0]

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index                   
                        
            if (iter + 1) % validation_frequency == 0:
                # compute loss on all training set
                train_losses = [validate_bytraindata_model(i)[0] for i in range(n_train_batches)]
                this_train_loss = np.mean(train_losses)
                
                # compute loss on validation set
                this_validation_loss = validate_model()[0]
                
                if (iter + 1) % 5 == 0 and process_display:                
                    print(
                        'epoch %i, minibatch %i/%i, training loss %f, validation loss %f ' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_train_loss,
                            this_validation_loss
                        )
                    )

                total_train_losses = [validate_bytraindata_model(i)[0]for i in range(n_train_batches)]
                total_train_losses = np.mean(total_train_losses)
                train_losses_record.append(total_train_losses)                    
                validate_losses_record.append(this_validation_loss)
                    
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    train_loss = this_train_loss
                    best_validation_loss = this_validation_loss                    
                    best_iter = iter
                                        
                    #Get Accuracy
                    
                    train_losses = [validate_bytraindata_model(i)[1]for i in range(n_train_batches)]
                    train_score_S = 1 - np.mean(train_losses)
                    train_losses = [validate_bytraindata_model(i)[2]for i in range(n_train_batches)]
                    train_score_T = 1 - np.mean(train_losses)
                    
                    validation_losses = validate_model()[1]
                    validation_score_S = 1 - np.mean(validation_losses)
                    validation_losses = validate_model()[2]
                    validation_score_T = 1 - np.mean(validation_losses)
                    
                    # test it on the test set
                    test_losses = test_model()[1]
                    test_score_S = 1 - np.mean(test_losses)
                    test_losses = test_model()[2]
                    test_score_T = 1 - np.mean(test_losses)
                    
                    trained_params_name = classifier.params_name()
                    trained_params_value = classifier.params_value()                    
                            
                    if process_display:                        
                        print(('     epoch %i, minibatch %i/%i, test accuracy of '
                           'best model: source domain :%f %%, target domain %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score_S * 100., test_score_T * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    if process_display:
        print(('Optimization complete. Best validation loss of %f '
           'obtained at iteration %i, with train loss %f \n'
           'train accuracy : source domain %f %%, target domain  %f %%\n'
           'validation accuracy : source domain %f %%, target domain  %f %%\n'
           'test accuracy : source domain %f %%, target domain  %f %%') %
          (best_validation_loss, best_iter + 1, train_loss, train_score_S * 100., train_score_T * 100.,
           validation_score_S * 100., validation_score_T * 100., test_score_S * 100., test_score_T * 100.))
    
    print('-------------------------------------------------------------------------')     
    
    #Converge curve
    index = range(len(train_losses_record))     
    title = 'Converge_Curve_%s' % (description)
    fts = (index, train_losses_record, index, validate_losses_record)    
    label = ('train loss', 'validation loss')
    color = [1, 2]
    marker = [0, 0]
    line = True    
    legend = True
    util.data2plot(title=title, fts=fts, label=label, color=color, marker=marker, line=line, legend=legend, plot_enable=process_display)                 
       
    print('-------------------------------------------------------------------------')
    
    trained_param = VANN_params()
    trained_param.update_value(trained_params_name, trained_params_value, struct)  
    
    num_S = train_ftd_source.get_value(borrow=True).shape[0]
    num_T = train_ftd_target.get_value(borrow=True).shape[0]
    feature_classifier = VANN(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [num_S, num_T],
        struct = struct,
        coef = coef,
        init_params = trained_param
    )     
    
    features_model = theano.function(
        inputs=[],
        outputs=feature_classifier.feature_outputs() + [feature_classifier.source_predict(), feature_classifier.target_predict()] +
        [feature_classifier.source_reconstruct(), feature_classifier.target_reconstruct()],
        givens={
            x_source: train_ftd_source,
            x_target: train_ftd_target
        }
    )        
    
    return features_model, test_model, trained_param    
        