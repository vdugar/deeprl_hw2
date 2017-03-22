from keras.models import model_from_json
import numpy as np
from policy import *
from preprocessors import *
from objectives import *
from core import *
import csv

"""Main DQN agent."""

class LinearDQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    STAGE_RANDOM_EXPLORE      = 1
    STAGE_TRAIN               = 2
    STAGE_TEST                = 3

    def __init__(self,
                 q_network,
                 preprocessor,
                 test_preprocessor,
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 params):
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.test_preprocessor = test_preprocessor
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.params = params

        self.target_network = None
        self.t = 0
        self.num_actions = 0
        self.policies = None
        self.prev_action = 0
        self.cumulative_reward = 0
        # self.loss_vector = [0] * 5000000

    def compile(self, optimizer, loss_func, wt_dir):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        
        # set up target Q-network
        if self.params['use_target']:
          model_json = self.q_network.to_json()
          weight_fn = wt_dir + 'weights_for_copy.h5'
          self.q_network.save_weights(weight_fn)
          self.target_network = model_from_json(model_json)
          self.target_network.load_weights(weight_fn)
          self.target_network.compile(loss=loss_func, optimizer=optimizer)

        self.q_network.compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state, q_network):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # hack to expand state dim to please keras
        state = np.expand_dims(state, 0)
        temp = q_network.predict_on_batch(state)
        return temp[0, :]

    def select_action(self, state, stage, preprocessor, q_net, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        
        # preprocess state
        state = preprocessor.process_state_for_network(state)

        # check if we're randomly exploring
        if stage == LinearDQNAgent.STAGE_RANDOM_EXPLORE:
          return self.policies['uniform'].select_action()

        # recover q_vals
        q_vals = self.calc_q_values(state, q_net)

        # choose policy depending on stage
        if stage == LinearDQNAgent.STAGE_TRAIN:
          # check if we should re-use the previous action
          if self.t % self.params['action_update_freq'] != 0:
            return self.prev_action
          
          policy = self.policies['decay_greedy']
        elif stage == LinearDQNAgent.STAGE_TEST:
          policy = self.policies['eps_greedy']

        # return action
        return policy.select_action(q_vals)

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        
        if self.t < self.num_burn_in:
          # still collecting initial samples. Do nothing.
          return None
        elif self.t % self.train_freq != 0:
          # don't update network on this step
          return None
        else:
          # sample transitions
          minibatch = self.preprocessor.process_batch(
            self.memory.sample(self.batch_size, [0]))

          # determine targets
          targets = []
          states = []
          for sample in minibatch:
            states.append(sample.state)
            pred = self.calc_q_values(sample.state, self.q_network)
            if sample.is_terminal:
              t = sample.reward
            else:
              if self.params['use_target']:
                target = self.target_network
              else:
                target = self.q_network
              t = sample.reward + self.gamma * np.max(self.calc_q_values(
                sample.next_state, target))
            
            # construct target vector for this sample. Since we only have
            # information corresponding to a single action, set the target
            # equal to the predicted q-value for all other actions. This way we
            # won't propagate spurious gradients.
            t_vect = np.float32(pred)
            t_vect[sample.action] = t
            targets.append(t_vect)

          # perform batch SGD
          loss = self.q_network.train_on_batch(np.array(states), np.array(targets))

          # check if we should update target network
          if self.params['use_target'] and self.t % self.target_update_freq == 0:
            # copy current network into target
            self.update_target_network()

        return loss

    def fit(self, env, env_test, num_iterations, log_dir, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        state = env.reset()
        self.preprocessor.reset()
        self.num_actions = env.action_space.n

        # set up policies
        self.policies = {
          'uniform': UniformRandomPolicy(self.num_actions),
          'greedy': GreedyPolicy(),
          'eps_greedy': GreedyEpsilonPolicy(self.params['epsilon'], self.num_actions),
          'decay_greedy': LinearDecayGreedyEpsilonPolicy(
            self.params['eps_start'], self.params['eps_end'], self.params['eps_num_steps'], 
          self.num_actions)
        }

        episode_time = 0
        is_terminal = False
        state_mem = self.preprocessor.process_state_for_memory(state)
        num_episodes = 0
        for i in range(0, num_iterations):
          self.t += 1
          episode_time += 1

          if is_terminal or ((max_episode_length is not None) and (episode_time > max_episode_length)):
            state = env.reset()
            self.preprocessor.reset()
            state_mem = self.preprocessor.process_state_for_memory(state)
            # print("Episode num: %d, Episode time: %d" % (num_episodes, episode_time))
            episode_time = 0
            num_episodes += 1
            self.cumulative_reward = 0

          if self.t < self.num_burn_in:
            stage = LinearDQNAgent.STAGE_RANDOM_EXPLORE
          else:
            stage = LinearDQNAgent.STAGE_TRAIN

          # get action
          action = self.select_action(state, stage, self.preprocessor, self.q_network)
          self.prev_action = action

          # take a step in the environment
          state, r, is_terminal, info = env.step(action)
          r_proc = self.preprocessor.process_reward(r)
          self.cumulative_reward += r_proc

          # add to replay memory
          prev_state_mem = state_mem
          state_mem = self.preprocessor.process_state_for_memory(state)
          self.memory.append(prev_state_mem, action, r_proc, state_mem, is_terminal)

          # update network
          loss = self.update_policy()

          # eval stuff
          # if (loss is not None) and (self.t % self.params['disp_loss_freq'] == 0):
          #   print("Time: %d, Loss = %f" % (self.t, loss))
          if self.t > self.num_burn_in and self.t % self.params['eval_freq'] == 0:
            self.evaluate(env_test, self.params['eval_episodes'])

          # if self.t % self.params['print_freq'] == 0:
          #   print("Iter num: %d" % self.t)

          # save weights
          if self.t % self.params['weight_save_freq'] == 0:
            fn = log_dir + "qnet_weights_" + str(self.t) + ".h5"
            self.q_network.save_weights(fn)

        return self.q_network

    def update_target_network(self):
      """ Updates the target network by setting it equal to the current Q-network """

      # num_layers = len(self.q_network.layers)
      # for i in range(num_layers):
      #   self.target_network.layers[i].set_weights(
      #     self.q_network.layers[i].get_weights())
      self.target_network.set_weights(self.q_network.get_weights())


    def evaluate(self, env, num_episodes, max_episode_length=10000):
      """Test your agent with a provided environment.
      
      You shouldn't update your network parameters here. Also if you
      have any layers that vary in behavior between train/test time
      (such as dropout or batch norm), you should set them to test.

      Basically run your policy on the environment and collect stats
      like cumulative reward, average episode length, etc.

      You can also call the render function here if you want to
      visually inspect your policy.
      """
      
      # run the policy for num episodes
      cum_reward = 0.
      total_episode_time = 0.
      stage = LinearDQNAgent.STAGE_TEST
      for i in range(num_episodes):
        # reset stuff
        self.test_preprocessor.reset()
        state = env.reset()
        episode_time = 0
        total_reward = 0
        is_terminal = False

        # play episode
        while not is_terminal:
          episode_time += 1

          # get action
          action = self.select_action(state, stage, self.test_preprocessor, self.q_network)

          # take a step in the environment
          state, r, is_terminal, info = env.step(action)
          r_proc = self.test_preprocessor.process_reward(r)
          total_reward += r_proc

        # update totals
        total_episode_time += episode_time
        cum_reward += total_reward

      print("%d,%f,%f" % (self.t, total_episode_time/num_episodes, cum_reward/num_episodes))
      # return (self.t, total_episode_time/num_episodes, cum_reward/num_episodes)

    def evaluate_with_render(self, env, max_episode_length, q_net):
      self.num_actions = env.action_space.n
      self.policies = {
        'uniform': UniformRandomPolicy(self.num_actions),
        'greedy': GreedyPolicy(),
        'eps_greedy': GreedyEpsilonPolicy(self.params['epsilon'], self.num_actions),
        'decay_greedy': LinearDecayGreedyEpsilonPolicy(
          self.params['eps_start'], self.params['eps_end'], self.params['eps_num_steps'], 
        self.num_actions)
      }

      cum_reward = 0.
      total_episode_time = 0.
      stage = LinearDQNAgent.STAGE_TEST
      self.test_preprocessor.reset()
      state = env.reset()
      episode_time = 0
      total_reward = 0
      total_reward_proc = 0
      is_terminal = False

      while not is_terminal:
        episode_time += 1
        action = self.select_action(state, stage, self.test_preprocessor, q_net)
        state, r, is_terminal, info = env.step(action)
        r_proc = self.test_preprocessor.process_reward(r)
        total_reward += r
        total_reward_proc += r_proc
        env.render()


      print("Episode time: %d" % episode_time)
      print("Score: %f, Clipped score: %f" % (total_reward, total_reward_proc))