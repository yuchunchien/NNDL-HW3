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
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn

import random
import numpy as np
import scipy


#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
def test_lenet(batch_size, n_epochs, learning_rate = 0.01):
   
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

    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # 4D output tensor is thus of shape (batch_size, 32, 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2)
    )

    # 4D output tensor is thus of shape (batch_size, 64, 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 32, 15, 15),
        filter_shape=(64, 32, 3, 3),  
        poolsize=(2, 2)
    )
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=64 * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ##### compute result
    train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
    
#Problem 2
def P2_lenet(batch_size, n_epochs, learning_rate = 0.01):
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[2]
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

    layer0_input = x.reshape((batch_size, 3, 32, 32))
    
    # 4D output tensor is thus of shape (batch_size, 32, 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2)
    )

    # 4D output tensor is thus of shape (batch_size, 64, 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 32, 15, 15),
        filter_shape=(64, 32, 3, 3),  
        poolsize=(2, 2)
    )
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=64 * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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


    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True    

    while (epoch < n_epochs) and (not done_looping):        
        ##### get original data
        train_set_x, train_set_y = datasets[0]
        
        ##### redefine train_model
        train_model_1 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )          
        
        
        epoch = epoch + 1
        # train with original data_set
        for minibatch_index in range(n_train_batches):

            iter = (2*epoch - 2) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model_1(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

                
        ##### retrain with train_sex_t
        train_set_x, train_set_y = datasets[0]
          
        ##### redefine train_model
        train_model_2 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
                
        # train with augmentation data_set       
        for minibatch_index in range(n_train_batches):

            iter = (2*epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model_2(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
   

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Best validation accuracy: %f%%.' % ((1.0-best_validation_loss) * 100.))
    print('Best test accuracy: %f%%.' % ((1.0-test_score) * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
#Problem 2.1
#Write a function to add translations
#input 3*32*32 ndarray, random1, random2
#image_out will move downward ran_1 leftward ran_2
def translate_image(t_images_in, ran_1, ran_2):
    # get numpy image_in
    images_in = t_images_in.get_value()
        
    for i in range(images_in.shape[0]):
        # get image
        xi = np.transpose(images_in[i].reshape(3,32,32),(1,2,0))
        
        # implement translation on 3 channel
        xi = np.roll(xi, ran_1, axis=0) # up down
        xi = np.roll(xi, ran_2, axis=1) # left right
 
        # save back
        images_in[i] = np.transpose(xi,(2,0,1)).flatten()
    
    t_images_in.set_value(images_in)       

#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(batch_size, n_epochs, learning_rate = 0.01):
   
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[2]
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

    layer0_input = x.reshape((batch_size, 3, 32, 32))
    
    # 4D output tensor is thus of shape (batch_size, 32, 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2)
    )

    # 4D output tensor is thus of shape (batch_size, 64, 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 32, 15, 15),
        filter_shape=(64, 32, 3, 3),  
        poolsize=(2, 2)
    )
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=64 * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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


    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True    

    while (epoch < n_epochs) and (not done_looping):        
        epoch = epoch + 1
        
        ##### implement translation
        train_set_x, train_set_y = datasets[0]
        ran_1 = int(random.uniform(-3,3))
        ran_2 = int(random.uniform(-3,3))
        translate_image(train_set_x, ran_1, ran_2)
                
        ##### redefine train_model
        train_model_2 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
                
        # train with augmentation data_set
        print('-----Training with augmented data-----')
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model_2(minibatch_index)   
        print('-----Training over-----')
        
        
        ##### get original data
        train_set_x, train_set_y = datasets[0]
        
        ##### redefine train_model
        train_model_1 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )          
        
        # train with original data_set
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model_1(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
                

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Best validation accuracy: %f%%.' % ((1.0-best_validation_loss) * 100.))
    print('Best test accuracy: %f%%.' % ((1.0-test_score) * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    
#Problem 2.2
#Write a function to add roatations
def rotate_image(t_images_in, ran, tackle_overflow=False):
    # get numpy image_in
    images_in = t_images_in.get_value()
        
    for i in range(images_in.shape[0]):
        # get image
        xi = np.transpose(images_in[i].reshape(3,32,32),(1,2,0))

        # rotate
        xi = scipy.ndimage.interpolation.rotate(xi, ran, reshape=False)
        
        # tackle overflow
        if tackle_overflow:
            for j in range(xi.shape[0]):
                for k in range(xi.shape[1]):
                    for l in range(xi.shape[2]):
                        if xi[j][k][l] > 1.0:
                            xi[j][k][l] = 1.0
                        elif xi[j][k][l] < 0.0:
                            xi[j][k][l] = 0.0
        
        # save back
        images_in[i] = np.transpose(xi,(2,0,1)).flatten()
    
    t_images_in.set_value(images_in)     

    
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(batch_size, n_epochs, learning_rate = 0.01):
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[2]
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

    layer0_input = x.reshape((batch_size, 3, 32, 32))
    
    # 4D output tensor is thus of shape (batch_size, 32, 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2)
    )

    # 4D output tensor is thus of shape (batch_size, 64, 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 32, 15, 15),
        filter_shape=(64, 32, 3, 3),  
        poolsize=(2, 2)
    )
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=64 * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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


    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True    

    while (epoch < n_epochs) and (not done_looping):        
        epoch = epoch + 1

        ##### add rotation
        train_set_x, train_set_y = datasets[0]
        ran = int(random.uniform(-3,3))
        rotate_image(train_set_x, ran)
        
        ##### redefine train_model
        train_model_2 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
                
        # train with augmentation data_set
        print('-----Training with augmented data-----')
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model_2(minibatch_index)   
        print('-----Training over-----')
        
        
        ##### get original data
        train_set_x, train_set_y = datasets[0]
        
        ##### redefine train_model
        train_model_1 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )          
        
        # train with original data_set
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model_1(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
                

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Best validation accuracy: %f%%.' % ((1.0-best_validation_loss) * 100.))
    print('Best test accuracy: %f%%.' % ((1.0-test_score) * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    
#Problem 2.3
#Write a function to flip images
def flip_image(t_images_in, ran):
    # if ran=1 do flip, if not don't
    if ran==0:
        return t_images_in
    
    # get numpy image_in
    images_in = t_images_in.get_value()
    
    for i in range(images_in.shape[0]): 
        # get xi
        xi = np.transpose(images_in[i].reshape(3,32,32),(1,2,0)) #images_in[i].reshape(3,32,32)
        
        # flip
        xi = np.fliplr(xi)
       
        # save bask
        images_in[i] = np.transpose(xi,(2,0,1)).flatten() #xi.flatten()  
    
    t_images_in.set_value(images_in)


#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(batch_size, n_epochs, learning_rate = 0.01):   
   
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[2]
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

    layer0_input = x.reshape((batch_size, 3, 32, 32))
    
    # 4D output tensor is thus of shape (batch_size, 32, 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2)
    )

    # 4D output tensor is thus of shape (batch_size, 64, 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 32, 15, 15),
        filter_shape=(64, 32, 3, 3),  
        poolsize=(2, 2)
    )
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=64 * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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


    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True    

    while (epoch < n_epochs) and (not done_looping):        
        epoch = epoch + 1
 
        ##### implement flip
        train_set_x, train_set_y = datasets[0]
        flip_image(train_set_x, 1)
        
        ##### redefine train_model
        train_model_2 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
                
        # train with augmentation data_set
        print('-----Training with augmented data-----')
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model_2(minibatch_index)   
        print('-----Training over-----')
        
        
        ##### get original data
        train_set_x, train_set_y = datasets[0]
        
        ##### redefine train_model
        train_model_1 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )          
        
        # train with original data_set
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model_1(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
                

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Best validation accuracy: %f%%.' % ((1.0-best_validation_loss) * 100.))
    print('Best test accuracy: %f%%.' % ((1.0-test_score) * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    
#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(t_images_in, ran, tackle_overflow=False):
    # get numpy image_in
    images_in = t_images_in.get_value()
    
    for i in range(images_in.shape[0]): 
        # get xi
        xi = np.transpose(images_in[i].reshape(3,32,32),(1,2,0))

        # add noise
        if(ran==1):
            xi += np.random.normal(loc=0.0, scale=0.1)
        elif(ran==0):
            xi += np.random.uniform(low=-0.1, high=0.1)
        else:
            print('error in add noise')

        # tackle overflow
        if tackle_overflow:
            for j in range(xi.shape[0]):
                for k in range(xi.shape[1]):
                    for l in range(xi.shape[2]):
                        if xi[j][k][l] > 1.0:
                            xi[j][k][l] = 1.0
                        elif xi[j][k][l] < 0.0:
                            xi[j][k][l] = 0.0
        
              
        # save bask
        images_in[i] = np.transpose(xi,(2,0,1)).flatten()
    
    t_images_in.set_value(images_in)
    
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(batch_size, n_epochs, learning_rate = 0.01):
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[2]
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

    layer0_input = x.reshape((batch_size, 3, 32, 32))
    
    # 4D output tensor is thus of shape (batch_size, 32, 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2)
    )

    # 4D output tensor is thus of shape (batch_size, 64, 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 32, 15, 15),
        filter_shape=(64, 32, 3, 3),  
        poolsize=(2, 2)
    )
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=64 * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )
    
    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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


    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True    

    while (epoch < n_epochs) and (not done_looping):        
        epoch = epoch + 1
        
        ##### add noise
        train_set_x, train_set_y = datasets[0]
        ran = int(random.uniform(0,2))
        noise_injection(train_set_x, ran)
                
        ##### redefine train_model
        train_model_2 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
                
        # train with augmentation data_set
        print('-----Training with augmented data-----')
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model_2(minibatch_index)   
        print('-----Training over-----')
        
        
        ##### get original data
        train_set_x, train_set_y = datasets[0]
        
        ##### redefine train_model
        train_model_1 = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )          
        
        # train with original data_set
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model_1(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
                

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Best validation accuracy: %f%%.' % ((1.0-best_validation_loss) * 100.))
    print('Best test accuracy: %f%%.' % ((1.0-test_score) * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
from my_lenet import my_lenet

def MY_lenet(batch_size=100, n_epochs=2000, learning_rate = 0.01, L2_reg = 0.0001, activation=T.tanh):
    my_lenet(batch_size=100, n_epochs=2000, learning_rate = 0.01, L2_reg = 0.0001, activation=T.tanh)

#Problem4
#Implement the convolutional neural network depicted in problem4 
from my_cnn import my_cnn

def MY_CNN(batch_size, n_epochs, learning_rate = 0.01, patience = 12000):
    my_cnn(batch_size, n_epochs, learning_rate = 0.01, patience = 12000)