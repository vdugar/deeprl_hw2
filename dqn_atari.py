#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute, Lambda)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
import keras

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.double_dqn import DoubleDQNAgent
from deeprl_hw2.linear_dqn import LinearDQNAgent
import gym
from deeprl_hw2.preprocessors import *
from deeprl_hw2.objectives import *
from deeprl_hw2.core import *
from gym import wrappers


def create_model_dqn(window, input_shape, num_actions,
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

    stack_size = window
    rows_image = input_shape[0]
    colms_image = input_shape[1]     


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

    return model

def create_model_dueling(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Dueling Q-network model.
    """      

    stack_size = window
    rows_image = input_shape[0]
    colms_image = input_shape[1]   

    S = Input(shape=(rows_image, colms_image, stack_size))
    l1 = Convolution2D (32 , 8 , 8, subsample = (4,4),
        input_shape=(rows_image,colms_image,stack_size), activation='relu')(S) 
    l2 = Convolution2D (64 , 4 , 4, subsample = (2,2), activation='relu')(l1)
    l3 = Convolution2D (64 , 3 , 3, subsample = (1,1), activation='relu')(l2)
    f1 = Flatten()(l3)
    v1 = Dense(512, activation='relu')(f1)
    a1 = Dense(512, activation='relu')(f1)

    v2 = Dense(1)(v1)
    a2 = Dense(num_actions)(a1)

    value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), 
        output_shape=(num_actions,))(v2)
    advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), 
        output_shape=(num_actions,))(a2)

    output = keras.layers.merge([value, advantage], mode='sum')

    model = Model(inputs = S, outputs = output)  
    
    return model

def create_model_linear(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103    

    stack_size = window
    rows_image = input_shape[0]
    colms_image = input_shape[1]     


    model = Sequential()
    model.add( Flatten(input_shape=(rows_image,colms_image,stack_size)) )
    model.add( Dense(512) )
    model.add( Activation( 'relu'))
    model.add(Dense(num_actions)) #no. of action determine    

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
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('-e', '--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('-n', '--network', default='dqn', help='Network Type')

    args = parser.parse_args()

    print args

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
        'action_update_freq': 1,
        'epsilon': 0.05,
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_num_steps': 1000000,
        'disp_loss_freq': 4000,
        'eval_freq': 10000,
        'weight_save_freq': 50000,
        'eval_episodes': 20,
        'print_freq': 100,
    }

    # create environment
    env = gym.make(args.env)
    env_test = gym.make(args.env)
    num_actions = env.action_space.n

    #create Tensor Flow Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # set up preprocessors
    atari_preprocessor = AtariPreprocessor((84,84))
    hist_preprocessor = HistoryPreprocessor(hist_length)
    preprocessor = PreprocessorSequence( (atari_preprocessor, hist_preprocessor) )

    test_atari_preprocessor = AtariPreprocessor((84,84))
    test_hist_preprocessor = HistoryPreprocessor(hist_length)
    test_preprocessor = PreprocessorSequence( (test_atari_preprocessor, test_hist_preprocessor) )
    print("Set up preprocessors")

    # set up replay memory
    memory = ReplayMemory(memory_size, memory_size)
    print("Set up memory")

    # get model and set up agent
    if args.network == 'dqn':
        q_network = create_model_dqn(hist_length, (84,84), num_actions)
        agent = DQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )
    elif args.network == 'ddqn':
        q_network = create_model_dqn(hist_length, (84,84), num_actions)
        agent = DoubleDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )  
    elif args.network == 'duel':
        q_network = create_model_dueling(hist_length, (84,84), num_actions)
        agent = DQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        ) 
    elif args.network == 'linear_naive':
        params['use_replay'] = False
        params['use_target'] = False
        q_network = create_model_linear(hist_length, (84,84), num_actions)
        
        # set params for no replay and no target 
        memory.resize(1)
        num_burn_in = 0

        agent = LinearDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )
    elif args.network == 'linear_soph':
        params['use_replay'] = True
        params['use_target'] = True
        q_network = create_model_linear(hist_length, (84,84), num_actions)

        agent = LinearDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )
    elif args.network == 'linear_double':
        q_network = create_model_linear(hist_length, (84,84), num_actions)

        agent = DoubleDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )       


    # Compile model in agent
    adam = Adam(lr=1e-4)
    agent.compile(adam, mean_huber_loss, args.output)
    print("Set up agent.")

    # fit model
    print("Fitting Model.")
    agent.fit(env, env_test, num_iterations, args.output, 1e4)

def eval_q_net():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('-e', '--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument(
        '-d', '--dir', default='atari-v0', help='Directory to read data from')
    parser.add_argument('-n', '--network', default='dqn', help='Network Type')

    args = parser.parse_args()

    print args

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
        'action_update_freq': 1,
        'epsilon': 0.05,
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_num_steps': 1000000,
        'disp_loss_freq': 4000,
        'eval_freq': 10000,
        'weight_save_freq': 50000,
        'eval_episodes': 20,
        'print_freq': 100,
    }

    # create environment
    env = gym.make(args.env)
    env_test = gym.make(args.env)
    vid_fn = lambda x: True
    env_test = wrappers.Monitor(env_test, args.dir+"monitor/", 
        video_callable = vid_fn,force=True)
    num_actions = env.action_space.n

    #create Tensor Flow Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # set up preprocessors
    atari_preprocessor = AtariPreprocessor((84,84))
    hist_preprocessor = HistoryPreprocessor(hist_length)
    preprocessor = PreprocessorSequence( (atari_preprocessor, hist_preprocessor) )

    test_atari_preprocessor = AtariPreprocessor((84,84))
    test_hist_preprocessor = HistoryPreprocessor(hist_length)
    test_preprocessor = PreprocessorSequence( (test_atari_preprocessor, test_hist_preprocessor) )
    print("Set up preprocessors")

    # set up replay memory
    memory = ReplayMemory(memory_size, memory_size)
    print("Set up memory")

    # get model and set up agent
    if args.network == 'dqn':
        q_network = create_model_dqn(hist_length, (84,84), num_actions)
        agent = DQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )
    elif args.network == 'ddqn':
        q_network = create_model_dqn(hist_length, (84,84), num_actions)
        agent = DoubleDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )  
    elif args.network == 'duel':
        q_network = create_model_dueling(hist_length, (84,84), num_actions)
        agent = DQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        ) 
    elif args.network == 'linear_naive':
        params['use_replay'] = False
        params['use_target'] = False
        q_network = create_model_linear(hist_length, (84,84), num_actions)
        
        # set params for no replay and no target 
        memory.resize(1)
        num_burn_in = 0

        agent = LinearDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )
    elif args.network == 'linear_soph':
        params['use_replay'] = True
        params['use_target'] = True
        q_network = create_model_linear(hist_length, (84,84), num_actions)

        agent = LinearDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )
    elif args.network == 'linear_double':
        q_network = create_model_linear(hist_length, (84,84), num_actions)

        agent = DoubleDQNAgent(
            q_network,
            preprocessor,
            test_preprocessor,
            memory,
            gamma,
            target_update_freq,
            num_burn_in,
            train_freq,
            batch_size,
            params
        )       


    # Compile model in agent
    adam = Adam(lr=1e-4)
    agent.compile(adam, mean_huber_loss, args.dir)
    print("Set up agent.")

    # Evaluate
    # special-case for first iteration
    agent.evaluate_with_render(env_test, 1, 10000, agent.q_network, 0)

    # for i in range(100000, 2000001, 100000):
    #     fn = args.dir + "qnet_weights_" + str(i) + ".h5"
    #     agent.q_network.load_weights(fn)
    #     agent.evaluate_with_render(env_test, 20, 10000, agent.q_network, i)

    # save videos
    steps = [600000, 1200000, 2000000]
    for step in steps:
        fn = args.dir + "qnet_weights_" + str(step) + ".h5"
        agent.q_network.load_weights(fn)
        agent.evaluate_with_render(env_test, 1, 10000, agent.q_network, step)

if __name__ == '__main__':
    main()
    # test_q_net()
    # eval_q_net()
