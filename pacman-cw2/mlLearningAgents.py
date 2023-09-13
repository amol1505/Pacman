# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

# Implemented by Amol Dhaliwal 


# Note: Since there are too many state-actions pairs for a large
# number of attempts to be fully covered, number of attempts is set to a minimum,
# which proves to be very beneficial

# No unneeded libraries are used
# All imports used are listed directly below

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.state = state
        self.legal_actions = []

    # Accessor functions for parameters
    def getState(self) -> GameState:
        return self.state

    def getLegalActions(self) -> list:
        return self.legal_actions

    def setLegalActions(self, actions: Directions):
        self.legal_actions = actions

    # Helper methods
    def __eq__(self, other):
        return hasattr(other, 'state') and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 1,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)

        # Count the number of games we have played
        self.episodesSoFar = 0

        # The variables below do not store for states that have no actions other than stopping

        # States seen so far except end condition states
        self.states = set()

        # Q-values for state-action pairs
        self.q_values = {}

        # Times action taken for state-action pairs
        self.n_values = {}

        self.last_state = None
        self.last_action = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.getScore() - startState.getScore()

    def initializeForNewState(self,
                              state: GameStateFeatures,
                              legal_actions: list):
        """
        Set initial Q-value and N-value for a new state encounter

        Args:
            state: A given state
            legal_actions: Given legal actions of the state
        """
        self.states.add(state)

        for action in legal_actions:
            self.q_values[(state, action)] = float(0.0)
            self.n_values[(state, action)] = 0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        else:
            return 0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        q_values_for_actions = []

        for action in state.getLegalActions():
            q_values_for_actions.append(self.getQValue(state, action))

        # For the case that state entered has no legal actions available
        if (len(q_values_for_actions) == 0):
            return self.getQValue(state, None)

        return max(q_values_for_actions)
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        self.q_values[(state, action)] = (self.getQValue(state, action) +
                                          self.alpha * (reward +
                                                        self.gamma * self.maxQValue(nextState) -
                                                        self.getQValue(state, action)))
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        self.n_values[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.n_values[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        if (counts <= self.maxAttempts) and (self.getEpisodesSoFar() != self.getNumTraining()):
            return utility + (abs(utility) * 2)

        return utility

    def getBestAction(self, state: GameStateFeatures) -> Directions:
        """
        Returns the action with the maximum expected utility
        out of legal actions of a given state using exploration function

        Args:
             state: A given state

        Returns:
            Action with maximum utility out of given actions
        """
        max_expected_utility = None
        max_expected_utility_action = None

        for action in state.getLegalActions():
            expected_utility = self.explorationFn(self.q_values[(state, action)],
                                                  self.n_values[(state, action)])

            if (max_expected_utility == None) or (expected_utility > max_expected_utility):
                max_expected_utility = expected_utility
                max_expected_utility_action = action

        return max_expected_utility_action

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()

        # Remove stopping action, as not needed
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Returns stopping action if no legal actions after removing stopping action
        if len(legal) == 0:
            return Directions.STOP

        # GameStateFeatures object will be used from here on instead of GameState
        current_state = GameStateFeatures(state)
        current_state.setLegalActions(legal)

        # Initialize for current state if current state is a new encounter
        if not (current_state in self.states):
            self.initializeForNewState(current_state, legal)

        # Learn using previous state and action, and current state and reward
        if self.last_state != None:
            self.learn(self.last_state,
                       self.last_action,
                       QLearnAgent.computeReward(self.last_state.getState(), current_state.getState()),
                       current_state)

        # Pick what action to take.
        current_action = self.getBestAction(current_state)

        # Update count of action taken in current state
        self.updateCount(current_state, current_action)

        # Update last values
        self.last_state = current_state
        self.last_action = current_action
        return current_action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Learn from win or loss
        if self.last_state != None:
            current_state = GameStateFeatures(state)

            self.learn(self.last_state,
                       self.last_action,
                       QLearnAgent.computeReward(self.last_state.getState(), current_state.getState()),
                       current_state)

        # Reset last values
        self.last_state = None
        self.last_action = None

        # Output end of game
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played
        self.incrementEpisodesSoFar()

        # Set learning parameters to zero when we are done with
        # the pre-set number of training episodes
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
