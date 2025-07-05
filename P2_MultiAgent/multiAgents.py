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


import math
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        preFood = currentGameState.getFood()
        total_score = 0
        closestFoodManDistance = 0  # closest food manhatan from new pacman, lower better
        closestGhostManDistance = 0  # closest ghost manhatan from new pacman, bigger better
        closestGhostIndex = 0
        foodDifference = len(preFood.asList()) - len(newFood.asList())

        first = True
        for food in newFood.asList():
            if manhattanDistance(food, newPos) < closestFoodManDistance or first:
                closestFoodManDistance = manhattanDistance(food, newPos)
                if first:
                    first = False

        first = True
        counter = 0
        for ghost in newGhostStates:
            if manhattanDistance(ghost.getPosition(), newPos) < closestGhostManDistance or first:
                closestGhostManDistance = manhattanDistance(ghost.getPosition(), newPos)
                closestGhostIndex = counter
                if first:
                    first = False
            counter += 1

        counter = 0
        closestGhostScared = False
        for remained_time in newScaredTimes:
            if remained_time > 0:
                if counter == closestGhostIndex:
                    closestGhostScared = True
                    break
            counter += 1

        if closestGhostScared:
            total_score += (100 * foodDifference)
            total_score -= (0.01 * closestFoodManDistance)
        else:
            total_score += (10 * foodDifference)
            total_score -= (0.1 * closestFoodManDistance)
            if closestGhostManDistance < 2:
                total_score -= (1000 * (closestGhostManDistance + 1))

        return total_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()

        def value(gameState_v, agentIndex, currentDepth):
            if gameState_v.isWin() or gameState_v.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState_v)
            else:
                first = True
                utility = 0
                if agentIndex == 0:
                    for legalAction in gameState_v.getLegalActions(agentIndex):
                        successorState = gameState_v.generateSuccessor(agentIndex, legalAction)
                        if first:
                            utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth)
                            first = False
                        else:
                            utility = max(utility, value(successorState, (agentIndex + 1) % numAgent, currentDepth))
                    return utility
                elif agentIndex != 0:
                    for legalAction in gameState_v.getLegalActions(agentIndex):
                        successorState = gameState_v.generateSuccessor(agentIndex, legalAction)
                        if first:
                            if (agentIndex + 1) % numAgent == 0:
                                utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth + 1)
                            else:
                                utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth)
                            first = False
                        else:
                            if (agentIndex + 1) % numAgent == 0:
                                utility = min(utility,
                                              value(successorState, (agentIndex + 1) % numAgent, currentDepth + 1))
                            else:
                                utility = min(utility, value(successorState, (agentIndex + 1) % numAgent, currentDepth))
                    return utility

        finalAction = ""
        first_v = True
        utility_v = 0
        for legalAction in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, legalAction)
            if first_v:
                utility_v = value(successorState, 1, 0)
                finalAction = legalAction
                first_v = False
            elif max(utility_v, value(successorState, 1, 0)) > utility_v:
                utility_v = max(utility_v, value(successorState, 1, 0))
                finalAction = legalAction

        return finalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()

        def value(gameState_v, agentIndex, currentDepth, alpha, beta):
            if gameState_v.isWin() or gameState_v.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState_v)
            else:
                first = True
                utility = 0
                if agentIndex == 0:
                    for legalAction in gameState_v.getLegalActions(agentIndex):
                        successorState = gameState_v.generateSuccessor(agentIndex, legalAction)
                        if first:
                            utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth, alpha, beta)
                            first = False
                        else:
                            utility = max(utility,
                                          value(successorState, (agentIndex + 1) % numAgent, currentDepth, alpha, beta))
                        if utility > beta:
                            return utility
                        alpha = max(alpha, utility)
                    return utility
                elif agentIndex != 0:
                    for legalAction in gameState_v.getLegalActions(agentIndex):
                        successorState = gameState_v.generateSuccessor(agentIndex, legalAction)
                        if first:
                            if (agentIndex + 1) % numAgent == 0:
                                utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth + 1, alpha,
                                                beta)
                            else:
                                utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth, alpha, beta)
                            first = False
                        else:
                            if (agentIndex + 1) % numAgent == 0:
                                utility = min(utility,
                                              value(successorState, (agentIndex + 1) % numAgent, currentDepth + 1,
                                                    alpha, beta))
                            else:
                                utility = min(utility,
                                              value(successorState, (agentIndex + 1) % numAgent, currentDepth, alpha,
                                                    beta))
                        if utility < alpha:
                            return utility
                        beta = min(beta, utility)
                    return utility

        finalAction = ""
        first_v = True
        utility_v = 0
        alpha_v = -math.inf
        beta_v = math.inf
        for legalAction in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, legalAction)
            if first_v:
                utility_v = value(successorState, 1, 0, alpha_v, beta_v)
                finalAction = legalAction
                first_v = False
            elif max(utility_v, value(successorState, 1, 0, alpha_v, beta_v)) > utility_v:
                utility_v = max(utility_v, value(successorState, 1, 0, alpha_v, beta_v))
                finalAction = legalAction
            if utility_v > beta_v:
                return finalAction
            alpha_v = max(alpha_v, utility_v)
        return finalAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()

        def value(gameState_v, agentIndex, currentDepth):
            if gameState_v.isWin() or gameState_v.isLose() or currentDepth >= self.depth:
                return self.evaluationFunction(gameState_v)
            else:
                first = True
                utility = 0
                if agentIndex == 0:
                    for legalAction in gameState_v.getLegalActions(agentIndex):
                        successorState = gameState_v.generateSuccessor(agentIndex, legalAction)
                        if first:
                            utility = value(successorState, (agentIndex + 1) % numAgent, currentDepth)
                            first = False
                        else:
                            utility = max(utility, value(successorState, (agentIndex + 1) % numAgent, currentDepth))
                    return utility
                elif agentIndex != 0:
                    for legalAction in gameState_v.getLegalActions(agentIndex):
                        successorState = gameState_v.generateSuccessor(agentIndex, legalAction)
                        if (agentIndex + 1) % numAgent == 0:
                            utility += value(successorState, (agentIndex + 1) % numAgent, currentDepth + 1)
                        else:
                            utility += value(successorState, (agentIndex + 1) % numAgent, currentDepth)
                    return utility / len(gameState_v.getLegalActions(agentIndex))

        finalAction = ""
        first_v = True
        utility_v = 0
        for legalAction in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, legalAction)
            if first_v:
                utility_v = value(successorState, 1, 0)
                finalAction = legalAction
                first_v = False
            elif max(utility_v, value(successorState, 1, 0)) > utility_v:
                utility_v = max(utility_v, value(successorState, 1, 0))
                finalAction = legalAction
        return finalAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    total_score = 0
    closestFoodManDistance = 0  # closest food manhatan from new pacman, lower better
    closestGhostManDistance = 0  # closest ghost manhatan from new pacman, bigger better

    first = True
    for food in foods.asList():
        if manhattanDistance(food, pacmanPosition) < closestFoodManDistance or first:
            closestFoodManDistance = manhattanDistance(food, pacmanPosition)
            if first:
                first = False

    first = True
    counter = 0
    for ghost_pos in ghostPositions:
        if manhattanDistance(ghost_pos, pacmanPosition) < closestGhostManDistance or first:
            closestGhostManDistance = manhattanDistance(ghost_pos, pacmanPosition)
            if first:
                first = False
        counter += 1

    counter = 0
    scaredGhostNumber = 0
    for remained_time in scaredTimers:
        if remained_time > 0:
            scaredGhostNumber += 1
        counter += 1

    total_score -= (1000 * len(currentGameState.getCapsules()))
    total_score -= (100 * len(foods.asList()))
    total_score -= (5 * closestFoodManDistance)
    total_score -= scaredGhostNumber
    if closestGhostManDistance < 2:
        if closestGhostManDistance == 0:
            total_score -= (10000 * (closestGhostManDistance + 1))
        else:
            total_score -= (10000 * closestGhostManDistance)

    return total_score

# Abbreviation
better = betterEvaluationFunction
