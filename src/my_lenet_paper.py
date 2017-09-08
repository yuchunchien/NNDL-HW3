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
from my_lenet_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, drop, DropoutHiddenLayer

import random
import numpy as np
import scipy

#Implement a convolutional neural network with the translation method for augmentation
def my_lenet(batch_size=250, n_epochs=2000, learning_rate = 0.01, L2_reg = 0.0001, activation=T.tanh, patience = 20000):
   
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
    
    # 4D output tensor is thus of shape (batch_size, 128, 32, 32)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(128, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=1
    )

    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(128, 128, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(2, 2),
        activation=activation,
        drop_p=1
    )

    # 4D output tensor is thus of shape (batch_size, 384, 16, 16)
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(384, 128, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=0.92
    )
    
    
    # 4D output tensor is thus of shape (batch_size, 384, 16, 16)
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, 384, 16, 16),
        filter_shape=(384, 384, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=1
    )
    
    ''' 
    # 4D output tensor is thus of shape (batch_size, 768, 8, 8)
    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 384, 8, 8),
        filter_shape=(768, 384, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=0.79
    )
     
    # 4D output tensor is thus of shape (batch_size, 768, 8, 8)
    layer5 = LeNetConvPoolLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, 768, 8, 8),
        filter_shape=(768, 768, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=1
    )
    
      
    # 4D output tensor is thus of shape (batch_size, 1280, 32, 32)
    layer6 = LeNetConvPoolLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, 768, 32, 32),
        filter_shape=(1280, 768, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=0.7
    )
    
    # 4D output tensor is thus of shape (batch_size, 1280, 32, 32)
    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer6.output,
        image_shape=(batch_size, 1280, 32, 32),
        filter_shape=(1280, 1280, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=1
    )
    
    # 4D output tensor is thus of shape (batch_size, 1920, 32, 32)
    layer8 = LeNetConvPoolLayer(
        rng,
        input=layer7.output,
        image_shape=(batch_size, 1280, 32, 32),
        filter_shape=(1920, 1280, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=0.6
    )
    
    # 4D output tensor is thus of shape (batch_size, 1920, 32, 32)
    layer9 = LeNetConvPoolLayer(
        rng,
        input=layer8.output,
        image_shape=(batch_size, 1920, 32, 32),
        filter_shape=(1920, 1920, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=1
    )
    
    
    # 4D output tensor is thus of shape (batch_size, 2688, 32, 32)
    layer10 = LeNetConvPoolLayer(
        rng,
        input=layer9.output,
        image_shape=(batch_size, 1920, 32, 32),
        filter_shape=(2688, 1920, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=0.5
    )
    
    # 4D output tensor is thus of shape (batch_size, 2688, 32, 32)
    layer11 = LeNetConvPoolLayer(
        rng,
        input=layer10.output,
        image_shape=(batch_size, 2688, 32, 32),
        filter_shape=(2688, 2688, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1),
        activation=activation,
        drop_p=1
    )
    '''

    layer12_input = layer3.output.flatten(2)
    
    layer12 = LogisticRegression(input=layer12_input, n_in=384*16*16, n_out=10)
    
    # the cost we minimize during training is the NLL of the model
    L2_sqr = (layer12.W ** 2).sum() + (layer3.W ** 2).sum() + (layer2.W ** 2).sum() + (layer1.W ** 2).sum() + (layer0.W ** 2).sum()
    
    cost = (layer12.negative_log_likelihood(y) + L2_reg * L2_sqr)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer12.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        [index],
        layer12.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer12.params + layer3.params + layer2.params + layer1.params + layer0.params
    

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    momentum =theano.shared(numpy.cast[theano.config.floatX](0.5), name='momentum')
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))
    
    
    ######################
    # TRAIN ACTUAL MODEL #
    ######################
    # early-stopping parameters
    # patience = 20000  look as this many examples regardless
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
        train_model_FLIP = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )    
                
        # train with augmentation data_set
        print('-----Training with augmented data (flip)-----')
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model_FLIP(minibatch_index)   
        print('-----Training over-----')
 
        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i
                             in range(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        if verbose:
            print('epoch %i, train augmented data (flip), validation error %f %%' %
                (epoch,
                 this_validation_loss * 100.))
        

        '''
        ##### add noise
        train_set_x, train_set_y = datasets[0]
        ran = int(random.uniform(0,2))
        noise_injection(train_set_x, ran)
        
        ##### redefine train_model
        train_model_NOISE = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                training_enabled: numpy.cast['int32'](1)
            }
        ) 
                
        # train with augmentation data_set
        print('-----Training with augmented data (noise)-----')
        for minibatch_index in range(n_train_batches):
            cost_ij = train_model_NOISE(minibatch_index)   
        print('-----Training over-----')
        
        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i
                             in range(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        if verbose:
            print('epoch %i, train augmented data (noise), validation error %f %%' %
                (epoch,
                 this_validation_loss * 100.))
        '''
        
        
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
