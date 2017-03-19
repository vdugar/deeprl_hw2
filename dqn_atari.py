#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
import gym
from deeprl_hw2.preprocessors import *
from deeprl_hw2.objectives import *
from deeprl_hw2.core import *


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """      

    #for now
    stack_size = window
    rows_image = input_shape[0]
    colms_image = input_shape[1]   
    # loss function is hardcoded to huber_loss
    # optimizer hardcoded to adam   


    model=Sequential()
    model.add(Convolution2D (32 , 8 , 8, subsample = (4,4),input_shape=(rows_image,colms_image,stack_size))) # subsample is the stride ( jump of the convolution filter)
    model.add( Activation( 'relu'))
    model.add(Convolution2D (64 , 4 , 4, subsample = (2,2)))
    model.add( Activation( 'relu'))
    model.add(Convolution2D (64 , 3 , 3, subsample = (1,1)))
    model.add( Activation( 'relu'))
    model.add( Flatten() )
    model.add( Dense(512))
    model.add( Activation( 'relu'))
    model.add(Dense(num_actions)) #no. of action determine    
    #changing to RMSProp
    #rms_opt=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)    
    #model.compile(loss='mse',optimizer=rms_opt)
    
    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    # parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    # parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    # parser.add_argument(
    #     '-o', '--output', default='atari-v0', help='Directory to save data to')
    # parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # args = parser.parse_args()
    # args.input_shape = tuple(args.input_shape)

    # args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    # define params
    gamma = 0.99
    target_update_freq = 10000
    num_burn_in = 50000
    train_freq= 4
    batch_size = 32
    hist_length = 4
    memory_size = 1000000
    num_iterations = 5000000
    params = {
        'action_update_freq': 4,
        'epsilon': 0.05,
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_num_steps': 1000000
    }

    # create environment
    env = gym.make('Enduro-v0')
    num_actions = env.action_space.n

    sess = tf.Session() #create Tensor Flow Session
    K.set_session(sess)

    # get model
    q_network = create_model(hist_length, (84,84), num_actions)
    print("Got model")
    print(q_network.layers[0].input_shape)

    # set up preprocessors
    atari_preprocessor = AtariPreprocessor((84,84))
    hist_preprocessor = HistoryPreprocessor(hist_length)
    preprocessor = PreprocessorSequence( (atari_preprocessor, hist_preprocessor) )
    print("Set up preprocessors")

    # set up replay memory
    memory = ReplayMemory(memory_size, memory_size)
    print("Set up memory")

    # set up agent
    agent = DQNAgent(
        q_network,
        preprocessor,
        memory,
        gamma,
        target_update_freq,
        num_burn_in,
        train_freq,
        batch_size,
        params
    )
    adam = Adam(lr=1e-6)
    agent.compile(adam, huber_loss)
    print("Set up agent.")

    # fit model
    print("Fitting Model.")
    agent.fit(env, num_iterations)

if __name__ == '__main__':
    main()
