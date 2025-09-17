# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattan_distance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def get_action(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.
        """
        # Collect legal moves and successor states
        legal_moves = gameState.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(gameState, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        return legal_moves[chosen_index]

    def evaluation_function(self, currentGameState, action):
        """
        Design a better evaluation function here.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        # CORRECTED: ghostState.scared_timer uses snake_case
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]

        return successor_game_state.get_score()

def score_evaluation_function(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.
    """

    def __init__(self, evalFn = 'score_evaluation_function', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """
    def get_action(self, gameState):
        
        def alpha_beta_search(state, agent_index, depth, alpha, beta):
            # Base case: terminal state or max depth reached
            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluationFunction(state), None

            is_pacman = (agent_index == 0)

            # Initialize best score based on agent type (MAX or MIN)
            if is_pacman:
                best_score = float('-inf')
            else:
                best_score = float('inf')

            best_action = None
            legal_actions = state.get_legal_actions(agent_index)

            for action in legal_actions:
                successor_state = state.generate_successor(agent_index, action)
                
                # Determine next agent and depth
                next_agent = (agent_index + 1) % state.get_num_agents()
                next_depth = depth
                if next_agent == 0:
                    next_depth += 1
                
                # Recursive call
                score, _ = alpha_beta_search(successor_state, next_agent, next_depth, alpha, beta)

                # Update best score and action
                if is_pacman:
                    if score > best_score:
                        best_score, best_action = score, action
                    alpha = max(alpha, best_score)
                    if best_score > beta: # Prune
                        break
                else:
                    if score < best_score:
                        best_score, best_action = score, action
                    beta = min(beta, best_score)
                    if best_score < alpha: # Prune
                        break

            return best_score, best_action

        # Initial call for Pacman
        _, action = alpha_beta_search(gameState, 0, 0, float('-inf'), float('inf'))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function.
    """
    pacman_pos = currentGameState.get_pacman_position()
    food_list = currentGameState.get_food().as_list()
    ghost_states = currentGameState.get_ghost_states()
    # CORRECTED: ghostState.scared_timer uses snake_case
    scared_timers = [ghostState.scared_timer for ghostState in ghost_states]

    # Start with the current score
    score = currentGameState.get_score()

    # Feature 1: Distance to the nearest food pellet
    if food_list:
        min_food_dist = min([manhattan_distance(pacman_pos, food) for food in food_list])
        score += 1.0 / min_food_dist

    # Feature 2: Number of remaining food pellets
    score -= 2 * len(food_list)

    # Feature 3: Ghost proximity (scared vs. active)
    for i, ghost in enumerate(ghost_states):
        ghost_pos = ghost.get_position()
        dist_to_ghost = manhattan_distance(pacman_pos, ghost_pos)

        if scared_timers[i] > 0:  # Ghost is scared
            if dist_to_ghost > 0:
                score += 200 / dist_to_ghost  # Bigger reward for being closer
        else:  # Ghost is NOT scared (dangerous)
            if dist_to_ghost < 2:
                score -= 500

    return score