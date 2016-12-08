from __future__ import print_function

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

################################################################################################################
################################################################################################################


'''Layer Definition'''
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid, name=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=name+'_W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name+'_b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b        
        
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        # parameters of the model
        self.params = [self.W, self.b]
        #self.params = [(name+'_W', self.W), (name+'_b', self.b)]


class ShareHiddenLayer(object):
    def __init__(self, rng, input_source, input_target, n_in, n_out,
                 W=None, b=None, activation=T.nnet.sigmoid, name=''):

        self.HL_source = HiddenLayer(
            rng=rng,
            input=input_source,
            n_in=n_in,
            n_out=n_out,
            W = W,
            b = b,
            activation=activation,
            name=name
        )
       
        self.HL_target = HiddenLayer(
            rng=rng,
            input=input_target,
            n_in=n_in,
            n_out=n_out,
            W = self.HL_source.params[0],
            b = self.HL_source.params[1],
            activation=activation,
            name=name
        )
        
        self.output_source = self.HL_source.output        
        self.output_target = self.HL_target.output
        
        # parameters of the model
        self.params = self.HL_source.params

        
class GaussianSampleLayer(object):
    def __init__(self, mu, log_sigma, n_in, batch_size):
        '''
        This layer is presenting the gaussian sampling process of Stochastic Gradient Variational Bayes(SVGB)

        :type mu: theano.tensor.dmatrix
        :param mu: a symbolic tensor of shape (n_batch, n_in), means the sample mean(mu)
        
        :type log_sigma: theano.tensor.dmatrix
        :param log_sigma: a symbolic tensor of shape (n_batch, n_in), means the log-variance(log_sigma). 
                          here using diagonal variance, so a data has a row variance.
        '''
        seed = 42
        '''
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
        '''
        srng = T.shared_randomstreams.RandomStreams(seed=seed)

        epsilon = srng.normal((batch_size, n_in))        
        #self.mu = mu
        #self.log_sigma = log_sigma
        #epsilon = np.asarray(rng.normal(size=(batch_size,n_in)), dtype=theano.config.floatX)
        self.output = mu + T.exp(0.5*log_sigma) * epsilon

class CatSampleLayer(object):
    def __init__(self, pi, n_in, batch_size):  
        '''
        This layer is presenting the categorical distribution sampling process of Stochastic Gradient Variational Bayes(SVGB)

        :type pi: theano.tensor.dmatrix
        :param pi: a symbolic tensor of shape (n_batch, n_in), means the probability of each category      
        '''
        
        seed = 42
        srng = T.shared_randomstreams.RandomStreams(seed=seed)
        #generate standard Gumbel distribution from uniform distribution
        epsilon = srng.uniform((batch_size, n_in))     
        
        c = 0.01                
        gamma = T.log(pi + c) + epsilon
        
        self.output = T.eq(gamma / T.max(gamma), T.ones((batch_size, n_in)))
        
################################################################################################################
################################################################################################################ 
        
class Adam_update(object):
    def __init__(self, cost, params, a = 0.001, b1 = 0.9, b2 = 0.999, e = 1e-8):
        '''
        Still in Testing
        
        This class deal with the adam gradient. Given the cost and parameters, it will create the adam update function as class field.
        
        type cost: theano.tensor.dscalar
        param cost: the objective cost function, should be scalar
        
        '''    
        one = T.constant(1)
        
        #calculate standard gradient
        gparams = [T.grad(cost, param) for param in params]
                   
        #record the number of learning step        
        t_step = theano.shared(value=np.float32(1), name='time_step', borrow=True)
        
        #initialize theano tensor variable for momnets
        n = len(gparams)
        self.m = []
        self.v = []        
        for i in range(n):
            dim = params[i].get_value(borrow=True).shape
            zeros_value = np.zeros(dim, dtype=theano.config.floatX)
            self.m.append( theano.shared(value=zeros_value, name=params[i].name+'_m', borrow=True) )
            self.v.append( theano.shared(value=zeros_value, name=params[i].name+'_v', borrow=True) )
        
        #create updating function             
        b1_c = (one -b1**t_step)
        b2_c = (one -b2**t_step)
        bc = T.sqrt(b2_c) / b1_c 
        self.new_m = []
        self.new_v = []        
        self.adam_gparams = []
        for i in range(n):
            self.new_m.append( (b1*self.m[i] + (one -b1)*gparams[i]) ) 
            self.new_v.append( (b2*self.v[i] + (one -b2)*gparams[i]**2) )        
            self.adam_gparams.append( ( bc*(self.new_m[i])/(T.sqrt(self.new_v[i])+e) ) )
        
        self.updates = []        
        for i in range(n):          
            self.updates.append( (self.m[i], self.new_m[i]) )
            self.updates.append( (self.v[i], self.new_v[i]) )            
            self.updates.append( (params[i], params[i] - a * self.adam_gparams[i]) )
            #self.updates.append( (params[i], params[i] - a*gparams[i] ) )
        self.updates.append( (t_step, t_step + 1) )
        
        self.t_step = t_step
        self.gparams = gparams
        self.b1_c = b1_c
        self.b2_c = b2_c
        
def adam(loss, all_params, all_learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1-1e-7):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    :parameters:
        - loss : Theano expression
            specifying loss
        - all_params : list of theano.tensors
            Gradients are calculated w.r.t. tensors in all_parameters
        - all_learning_Rate : array float, learning rate for each params
        - beta1 : float
            Exponentioal decay rate on 1. moment of gradients
        - beta2 : float
            Exponentioal decay rate on 2. moment of gradients
        - epsilon : float
            For numerical stability
        - gamma: float
            Decay on first moment running average coefficient
        - Returns: list of update rules
    """

    updates = []
    all_grads = theano.grad(loss, all_params)

    i = theano.shared(np.float32(1))  # HOW to init scalar shared?
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED

    for param_i, g, r in zip(all_params, all_grads, all_learning_rate):
        m = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t TO use beta1_t
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (r * (T.sqrt(fix2) / fix1) * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t) )
    updates.append((i, i_t))
    return updates        
        
################################################################################################################
################################################################################################################        
        
'''
Data/Parameter Structure Definition
Here NN parameter refer to Weight, bias. NN structure refer to hidden dimension and accroding activation function 
'''

class NN_struct:
    def __init__(self):
        self.layer_dim = []
        self.activation = []
        self.learning_rate = []
        self.decay = []

#Gaussian MLP        
class GMLP_struct:
    def __init__(self):
        #mu sigma share layer
        self.share = NN_struct()        
        #mu part
        self.mu = NN_struct()        
        #sigma part
        self.sigma = NN_struct()
        
'''
Neural Network Block Definition
here refer to complete Neural Network system block with fixed hidden layer number
'''       
################################################################################################################
################################################################################################################         
class NN_Block:
    def __init__(self, rng, input_source, input_target, struct, params = None, name=''):        
        
        l = len(struct.activation) 
                
        self.Layer_group = []         
        self.params = []        
        self.learning_rate = []
        self.decay = []        
        
        if l == 0:
            self.output_source = input_source
            self.output_target = input_target
            print("%s is empty, link input to output directly" % (name))
            return        
        
        print("%s is constructed with hidden layer number %i" % (name, l-1))
                                         
        if params == None:
            init_W = None
            init_b = None
        else:
            init_W = params[0]
            init_b = params[1]            
        layer_name = '%s_L%i' % (name, 1)
                
        tmp = ShareHiddenLayer(
            rng=rng,
            input_source=input_source,
            input_target=input_target,
            n_in=struct.layer_dim[0],
            n_out=struct.layer_dim[1],
            W = init_W,
            b = init_b,            
            activation=struct.activation[0],
            name=layer_name
        ) 
        
        self.Layer_group.append(tmp)
        self.params = tmp.params
        self.learning_rate.append(theano.shared(value=np.float32(struct.learning_rate[0]), name=layer_name+'_lr', borrow=True))
        self.decay.append(struct.decay[0])
        
        for i in range(l-1):
            j = i + 1
            if params == None:
                init_W = None
                init_b = None
            else:
                init_W = params[j*2]
                init_b = params[j*2+1]
                
            layer_name = '%s_L%i' % (name, j)
            
            tmp = ShareHiddenLayer(
                rng=rng,
                input_source=self.Layer_group[-1].output_source,
                input_target=self.Layer_group[-1].output_target,
                n_in=struct.layer_dim[j],
                n_out=struct.layer_dim[j+1],
                W = init_W,
                b = init_b,             
                activation=struct.activation[j],
                name=layer_name
            )                
            self.Layer_group.append(tmp)
            self.params = self.params + tmp.params
            self.learning_rate.append(theano.shared(value=np.float32(struct.learning_rate[j]), name=layer_name+'_lr', borrow=True))
            self.decay.append(struct.decay[j])            
                        
        self.output_source = self.Layer_group[-1].output_source
        self.output_target = self.Layer_group[-1].output_target
        
################################################################################################################
################################################################################################################  
class Gaussian_MLP:
    def __init__(self, rng, input_source, input_target, struct, batch_size, params = None, name=''):
        if params == None:
            params = [None, None, None]
                
        #share_block                
        self.share_block = NN_Block(
            rng=rng,            
            input_source=input_source,
            input_target=input_target,
            struct = struct.share,
            params = params[0],
            name=name+'_share'
        )              
                            
        #mu output block
        self.mu_block = NN_Block(
            rng=rng,            
            input_source=self.share_block.output_source,
            input_target=self.share_block.output_target,
            struct = struct.mu,
            params = params[1],
            name=name+'_mu'
        )   

        #sigma output block
        self.sigma_block = NN_Block(
            rng=rng,            
            input_source=self.share_block.output_source,
            input_target=self.share_block.output_target,
            struct = struct.sigma,
            params = params[2],
            name=name+'_sigma'
        )          
        
        self.S_mu = self.mu_block.output_source
        self.S_log_sigma = self.sigma_block.output_source
        self.S_sigma = T.exp(self.S_log_sigma)
        self.T_mu = self.mu_block.output_target
        self.T_log_sigma = self.sigma_block.output_target
        self.T_sigma = T.exp(self.T_log_sigma)
        
        self.GSL_source = GaussianSampleLayer(
            mu=self.S_mu,
            log_sigma = self.S_log_sigma,
            n_in = struct.mu.layer_dim[-1],
            batch_size = batch_size[0]
        )        
        
        self.GSL_target = GaussianSampleLayer(
            mu=self.T_mu,
            log_sigma = self.T_log_sigma,
            n_in = struct.mu.layer_dim[-1],
            batch_size = batch_size[1]
        )         
                        
        self.S_output = self.GSL_source.output
        self.T_output = self.GSL_target.output        
        
        self.params = self.share_block.params + self.mu_block.params + self.sigma_block.params
        self.learning_rate = self.share_block.learning_rate + self.mu_block.learning_rate + self.sigma_block.learning_rate
        self.decay = self.share_block.decay + self.mu_block.decay + self.sigma_block.decay