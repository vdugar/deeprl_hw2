"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor
from collections import deque
from deeprl_hw2.core import Sample


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):

        # Create DeQue data type and fill up zeros for lenght(history_length)
        self.history_length = history_length
        self.d_nw_flt = deque()     # network history
        self.d_mem_int = deque()    # memory history 
        
        # create empty queue with zero images     
        for i in range(history_length):            
            self.d_nw_flt.append(np.zeros((84,84), dtype=np.float32))        
            self.d_mem_int.append(np.zeros((84,84), dtype=np.uint8) )

    def process_state_for_network(self, state, **kwargs):
        #remove element from the que on the left.
        self.d_nw_flt.popleft()

        # append current state
        self.d_nw_flt.append(state)

        # construct sequence
        image_block_nw = self.d_nw_flt[0]
        for i in range(1, self.history_length):
            image_block_nw = np.dstack( (image_block_nw, self.d_nw_flt[i]) )
        
        return image_block_nw
        

    def process_state_for_memory(self, state, **kwargs):
        #remove element from the que on the left.
        self.d_mem_int.popleft()

        # append current state
        self.d_mem_int.append(state)

        # construct sequence
        image_block_nw = self.d_mem_int[0]
        for i in range(1, self.history_length):
            image_block_nw = np.dstack( (image_block_nw, self.d_mem_int[i]) )
        
        return image_block_nw


    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.d_nw_flt.clear()
        self.d_mem_int.clear()
        for i in range(self.history_length):            
            self.d_nw_flt.append(np.zeros((84,84), dtype=np.float32))        
            self.d_mem_int.append(np.zeros((84,84), dtype=np.uint8) )

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size = (84,84)):
        self.new_size = new_size
        self.downsample_size = (110, 84)

    def process_state_for_memory(self, state, **kwargs):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        img = Image.fromarray(np.uint8(state))
        img = img.convert('L').resize(self.downsample_size).crop((0, 0, 84, 84))
        return np.array(img, dtype=np.uint8)


    def process_state_for_network(self, state, **kwargs):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.

        """
        img = Image.fromarray(np.uint8(state))
        img = img.convert('L').resize(self.downsample_size).crop((0, 0, 84, 84))
        return np.array(img, dtype=np.float32) / 255.

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.sign(reward)

    def reset(self):
        pass

class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process_state_for_network(self, frame, **kwargs):
        for p in self.preprocessors:
            frame = p.process_state_for_network(frame, **kwargs)

        return frame

    def process_state_for_memory(self, frame, **kwargs):
        for p in self.preprocessors:
            frame = p.process_state_for_memory(frame, **kwargs)

        return frame
        
    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        new_samples = []
        for sample in samples:
            s = Sample(np.float32(sample.state) / 255., sample.action, sample.reward, np.float32(sample.next_state) / 255., sample.is_terminal)
            new_samples.append(s)

        return new_samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.sign(reward)

    def reset(self):
        for p in self.preprocessors:
            p.reset()

