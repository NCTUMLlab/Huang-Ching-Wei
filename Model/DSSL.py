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

class DSSL_struct(object):
    def __init__(self):
        self.encoder = nn.NN_struct()
        self.classifier = nn.NN_struct()        
        self.decoder = nn.NN_struct()

class DSSL_coef(object):
    def __init__(self, alpha = 1, beta = 1, D = 50, optimize = 'Adam_update'):
        self.alpha = alpha
        self.beta = beta                       # weight for MMD
        self.D = D                             # number of random feature for fast MMD
        self.optimize = optimize               # option of optimization

class DSSL_params(object):
    def __init__(self):
        self.EC_params=None
        self.CF_params=None
        self.DE_params=None
        
    def update_value(self, params_name, params_value, struct):
        
        params = [ theano.shared(value=value, name=name, borrow=True)
            for value, name in zip(params_value, params_name)]
        
        self.EC_params = []
        self.CF_params = []
        self.DE_params = []
        
        i = 0;
        for j in range(len(struct.encoder.activation)):
            self.EC_params.append(params[i])
            i = i+1
            self.EC_params.append(params[i])
            i = i+1

        for j in range(len(struct.classifier.activation)):
            self.CF_params.append(params[i])
            i = i+1        
            self.CF_params.append(params[i])
            i = i+1 
            
        for j in range(len(struct.decoder.activation)):
            self.DE_params.append(params[i])
            i = i+1        
            self.DE_params.append(params[i])
            i = i+1             
    
    def update_symbol(self, params, struct): 
        
        self.EC_params = []
        self.CF_params = []
        self.DE_params = []
        
        i = 0;
        for j in range(len(struct.encoder.activation)):
            self.EC_params.append(params[i])
            i = i+1
            self.EC_params.append(params[i])
            i = i+1

        for j in range(len(struct.classifier.activation)):
            self.CF_params.append(params[i])
            i = i+1        
            self.CF_params.append(params[i])
            i = i+1 
            
        for j in range(len(struct.decoder.activation)):
            self.DE_params.append(params[i])
            i = i+1        
            self.DE_params.append(params[i])
            i = i+1             

################################################################################################################          
        
'''Model Definition/Construct'''

class DSSL(object):   
    """
    The semi-supervised model Domain-Adversial Variational Autoencoder
    To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
    The domain label s will constuct inside this class
    For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
    Encoder refer to Encoder DSSL, Decoder refer to Decoder DSSL    
    """

    def __init__(self, rng, input_source, input_target, label_source, batch_size, struct, coef, train = False, init_params=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input_source: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
        
        :type input_target: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)        

        :type xxx_struct: class DSSL_struct
        :param xxx_strucat: define the structure of each DSSL
        """
        
        self.struct = struct
        encoder_struct = struct.encoder
        classifier_struct = struct.classifier
        decoder_struct = struct.decoder
        
        beta = coef.beta
        alpha = coef.alpha
        D = coef.D
        optimize = coef.optimize
        
        if init_params == None:
            init_params = DSSL_params()            
        
        #------------------------------------------------------------------------
        #Encoder Neural Network                            
        self.Encoder = nn.NN_Block(
            rng=rng,            
            input_source=input_source,
            input_target=input_target,
            struct = encoder_struct,
            params = init_params.EC_params,
            name='Encoder'
        )        
        
        self.z_S = self.Encoder.output_source
        self.z_T = self.Encoder.output_target       
        
        z_dim = encoder_struct.layer_dim[-1]
        
        self.Encoder_params = self.Encoder.params
        self.Encoder_learning_rate = self.Encoder.learning_rate     
        self.Encoder_decay = self.Encoder.decay       

        #------------------------------------------------------------------------
        #Decoder Neural Network                            
        self.Classifier = nn.NN_Block(
            rng=rng,            
            input_source=self.z_S,
            input_target=self.z_T,
            struct = classifier_struct,
            params = init_params.CF_params,
            name='Classifier'
        )        
        
        self.y_S = self.Classifier.output_source
        self.y_T = self.Classifier.output_target       
        
        self.Classifier_params = self.Classifier.params  
        self.Classifier_learning_rate = self.Classifier.learning_rate
        self.Classifier_decay = self.Classifier.decay
        
        self.Decoder = nn.NN_Block(
            rng=rng,            
            input_source=self.z_S,
            input_target=self.z_T,
            struct = decoder_struct,
            params = init_params.DE_params,
            name='Decoder'
        )        
        
        self.x_hat_S = self.Decoder.output_source
        self.x_hat_T = self.Decoder.output_target       
        
        self.Decoder_params = self.Decoder.params  
        self.Decoder_learning_rate = self.Decoder.learning_rate
        self.Decoder_decay = self.Decoder.decay

        #------------------------------------------------------------------------
        # Error Function Set                
        # Classification only source data-----------
        threshold = 0.0000001     
        self.Error_source = T.mean(T.sum(- label_source * T.log( T.maximum(self.y_S, threshold)), axis=1))
        
        # MMD betwween s, x using gaussian kernel-----------
        #self.MMD = MMD(self.zy_S, self.zy_T, batch_size)
        self.MMD = er.MMDEstimator(rng, self.z_S, self.z_T, z_dim, batch_size, D)  
                
        #Reconstruct error
        self.reconstruct_source = T.mean( (input_source - self.x_hat_S).norm(2, axis = 1) )
        self.reconstruct_target = T.mean( (input_target - self.x_hat_T).norm(2, axis = 1) )
        self.reconstruct = self.reconstruct_source + self.reconstruct_target    
            
        #Cost function
        self.cost = self.Error_source + self.MMD * beta + self.reconstruct * alpha
        
        # the parameters of the model
        self.params = self.Encoder_params + self.Classifier_params + self.Decoder_params
        self.learning_rate= self.Encoder_learning_rate + self.Classifier_learning_rate + self.Decoder_learning_rate
        self.decay= self.Encoder_decay + self.Classifier_decay + self.Decoder_decay

        
        if optimize == 'Adam_update' and train:
            #Adam update function
            self.updates = nn.adam(
                loss = self.cost,
                all_params = self.params,
                all_learning_rate= self.learning_rate
            )        
        elif optimize == 'SGD' and train:
            #Standard update function
            gparams = [T.grad(self.cost, param) for param in self.params]
        
            self.params_updates = [
                (param, param - learning_rate * gparam)
                for param, gparam, learning_rate in zip(self.params, gparams, self.learning_rate)
            ]        

            self.learning_rate_update = [
                (learning_rate, learning_rate * decay)
                for learning_rate, decay in zip(self.learning_rate, self.decay)
            ]
        
            self.updates = self.params_updates + self.learning_rate_update
                
        # keep track of model input
        self.input_source = input_source
        self.input_target = input_target            
        
        #Predict Label
        self.y_pred_source = T.argmax(self.y_S, axis=1)
        self.y_pred_target = T.argmax(self.y_T, axis=1)
                
    def feature_outputs(self):
        return [self.z_S, self.z_T]          

    def source_predict_raw(self):
        return self.y_S
    
    def target_predict_raw(self):
        return self.y_T      
        
    def source_predict(self):
        return self.y_pred_source
    
    def target_predict(self):
        return self.y_pred_target 
        
    def source_errors(self, y):
        #Classification Error
        return T.mean(T.neq(self.y_pred_source, T.argmax(y, axis=1)))

    def target_errors(self, y):
        #Classification Error
        return T.mean(T.neq(self.y_pred_target, T.argmax(y, axis=1)))
    
    def params_name(self):
        params_name = ( param.name for param in self.params)
        
        return params_name
    
    def params_value(self):
        params_value = ( param.get_value() for param in self.params)
        
        return params_value
    
    def params_symbol(self):
        tmp = DSSL_params()
        tmp.update_symbol(self.params, self.struct)
        return tmp   
    
def DSSL_training(source_data, target_data, n_train_batches, n_epochs, struct, coef, description, process_display=True):
             
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
    
    #x_dim = train_ftd_source.get_value(borrow=True).shape[1]
    #y_dim = train_labeld_target.get_value(borrow=True).shape[1]      
       
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
               
    # construct the DSSL class
    classifier = DSSL(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [batch_size_S, batch_size_T],
        struct = struct,
        coef = coef,
        train = True
    )    

    validate_classifier = DSSL(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [validate_S_size, validate_T_size],
        struct = struct,
        coef = coef,
        init_params = classifier.params_symbol()
    )    

    test_classifier = DSSL(
        rng=rng,
        input_source = x_source,
        input_target = x_target,
        label_source = y_source,
        batch_size = [test_S_size, test_T_size],
        struct = struct,
        coef = coef,
        init_params = classifier.params_symbol()
    )    
        
    #update function
    updates = classifier.updates

            
    test_model = theano.function(
        inputs=[],
        outputs=[test_classifier.cost, test_classifier.source_errors(y_source), test_classifier.target_errors(y_target), 
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
        outputs=[validate_classifier.cost, validate_classifier.source_errors(y_source), validate_classifier.target_errors(y_target), 
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
        outputs=[classifier.cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict(), classifier.target_predict()],
        givens={
            x_source: train_ftd_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            y_source: train_labeld_source[index * batch_size_S : (index + 1) * batch_size_S, :],
            x_target: train_ftd_target[index * batch_size_T : (index + 1) * batch_size_T, :],
            y_target: train_labeld_target[index * batch_size_T : (index + 1) * batch_size_T, :]            
        }       
    )     
    
    train_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        updates=updates,
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

            minibatch_avg_cost = train_model(minibatch_index)[0]  

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
    
    trained_param = DSSL_params()
    trained_param.update_value(trained_params_name, trained_params_value, struct)
    
    num_S = train_ftd_source.get_value(borrow=True).shape[0]
    num_T = train_ftd_target.get_value(borrow=True).shape[0]
    feature_classifier = DSSL(
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
        outputs=feature_classifier.feature_outputs() + [feature_classifier.source_predict(), feature_classifier.target_predict()],
        givens={
            x_source: train_ftd_source,
            x_target: train_ftd_target
        }
    )    
    
    return features_model, test_model, trained_param