from keras.models import model_from_json
import numpy as np
from policy import *
from preprocessors import *
from objectives import *
from core import *

"""Main DQN agent."""

class DQNAgent:
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
                 memory,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 params):
        self.q_network = q_network
        self.preprocessor = preprocessor
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


    def compile(self, optimizer, loss_func):
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
        model_json = self.q_network.to_json()
        self.q_network.save_weights('weights_for_copy.h5')
        self.target_network = model_from_json(model_string)
        self.target_network.load_weights('weights_for_copy.h5')

        self.q_network.compile(loss=loss_func, optimizer=optimizer)
        self.target_network.compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state, q_network):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        
        return q_network.predict(state)

    def select_action(self, state, stage, **kwargs):
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
        state = self.preprocessor.process_state_for_network(state)

        # check if we're randomly exploring
        if stage == STAGE_RANDOM_EXPLORE:
          return self.policies['uniform'].select_action()

        # check if we should re-use the previous action
        if self.t % self.params['action_update_freq'] != 0:
          return self.prev_action

        # recover q_vals
        q_vals = self.calc_q_values(state)

        # choose policy depending on stage
        if stage == STAGE_TRAIN:
          policy = self.policies['decay_greedy']
        elif stage == STAGE_TEST:
          policy == self.policies['eps_greedy']

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
            self.memory.sample(self.batch_size))

          # determine targets
          targets = []
          states = []
          for sample in minibatch:
            states.append(sample.state)
            pred = self.calc_q_values(sample.state, self.q_network)
            if sample.is_terminal:
              t = sample.reward
            else:
              t = sample.reward + self.gamma * max(self.calc_q_values(
                sample.next_state, self.target_network))
            
            # construct target vector for this sample. Since we only have
            # information corresponding to a single action, set the target
            # equal to the predicted q-value for all other actions. This way we
            # won't propagate spurious gradients.
            t_vect = np.float32(pred)
            t_vect[sample.action] = t
            targets.append(t_vect)

          # perform batch SGD
          loss = self.q_network.train_on_batch(states, target)

          # check if we should update target network
          if self.t % self.target_update_freq == 0:
            # copy current network into target
            self.update_target_network()

        return loss

    def fit(self, env, num_iterations, max_episode_length=None):
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
        for i in range(0, num_iterations):
          self.t += 1
          episode_time += 1
          if is_terminal or ((max_episode_length is not None) and (episode_time > max_episode_length)):
            state = env.reset()
            self.preprocessor.reset()
            state_mem = self.preprocessor.process_state_for_memory(state)
            episode_time = 0

          if self.t < self.num_burn_in:
            stage = STAGE_RANDOM_EXPLORE
          else:
            stage = STAGE_TRAIN

          # get action
          action = self.select_action(state, stage)
          self.prev_action = action

          # take a step in the environment
          state, r, is_terminal, info = env.step(action)
          r_proc = self.preprocessor.process_reward(r)

          # add to replay memory
          prev_state_mem = state_mem
          state_mem = self.process_state_for_memory(state)
          self.memory.append(prev_state_mem, action, r_proc, state_mem, is_terminal)

          # update network
          loss = self.update_policy()

    def update_target_network(self):
      """ Updates the target network by setting it equal to the current Q-network """

      num_layers = len(self.q_network.layers)
      for i in range(num_layers):
        self.target_network.layers[i].set_weights(
          self.q_network.layers[i].get_weights())


    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass
