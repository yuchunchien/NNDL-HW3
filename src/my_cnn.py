"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

from __future__ import print_function

import timeit
import inspect
import sys
import numpy
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.tensor.signal import pool

from hw3_utils import shared_dataset, load_data
from my_cnn_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn, UpSampleLayer, DropLayer

import numpy as np
import random
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Problem4
#Implement the convolutional neural network depicted in problem4 
def my_cnn(batch_size, n_epochs, learning_rate = 0.01, patience = 12000):
    
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
        
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches  //= batch_size
   
    
    rng = np.random.RandomState(23455)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
        
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    layerX_input = x.reshape((batch_size, 3, 32, 32))
    
    layerX = DropLayer(input=layerX_input)
   
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layerX.output,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),  
        poolsize=(2, 2)
    )
    # 4D output tensor is thus of shape (batch_size, 64, 16, 16)

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)
    
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),  
        poolsize=(2, 2)
    )
    # 4D output tensor is thus of shape (batch_size, 128, 8, 8)   
    
    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 256, 8, 8)    
    
    layer5 = UpSampleLayer(input=layer4.output)
    # 4D output tensor is thus of shape (batch_size, 256, 16, 16)  
    
    layer6 = LeNetConvPoolLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)      

    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer6.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)
    
    layer8 = UpSampleLayer(input=layer7.output+layer3.output_x)
    # 4D output tensor is thus of shape (batch_size, 128, 32, 32)
    
    layer9 = LeNetConvPoolLayer(
        rng,
        input=layer8.output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(64, 128, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
    
    layer10 = LeNetConvPoolLayer(
        rng,
        input=layer9.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
    
    layer11 = LeNetConvPoolLayer(
        rng,
        input=layer10.output+layer1.output_x,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3),  
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 3, 32, 32)  
    

    cost = layer11.ob_func(layerX_input)
 
 
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [],
        [layerX_input,layerX.output,layer11.output,cost],
        givens={
            x: test_set_x[0:100]
        }
    )
    
    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent    
    params = layer11.params + layer10.params + layer9.params + layer7.params + layer6.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    
   
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)    
    
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost, 
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    
    print('... training the model')   
    
    # early-stopping parameters
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_cost = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_cost = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_cost = numpy.mean(validation_cost)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation cost %f' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_cost))

                # if we got the best validation score until now
                if this_validation_cost < best_validation_cost:

                    # save best validation score and iteration number
                    best_validation_cost = this_validation_cost
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break
                
    TEST_MODEL_RESULT = test_model()
    GT_Images_T = TEST_MODEL_RESULT[0]
    Drop_Images_T = TEST_MODEL_RESULT[1] 
    Reconstructed_Images_T = TEST_MODEL_RESULT[2]
    cost_list = TEST_MODEL_RESULT[3] 
    

    # plot 8*3 images
    print("Ground Truth, Corrupted Images, and Recontructed Images:")
    f, axarr = plt.subplots(8,3,figsize=(20,20))
    for i in range(8):
        plt.axes(axarr[i,0])
        plt.imshow(np.transpose(GT_Images_T[i],(1,2,0)))
        
        plt.axes(axarr[i,1])
        plt.imshow(np.transpose(Drop_Images_T[i],(1,2,0))) 
        
        plt.axes(axarr[i,2])
        plt.imshow(np.transpose(Reconstructed_Images_T[i],(1,2,0)))

        
    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation cost %f obtained at iteration %i, ' %
          (best_validation_cost, best_iter + 1))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

