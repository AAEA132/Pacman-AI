# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
            tempValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    tempValues[state] = 0
                else:
                    possibleActions = self.mdp.getPossibleActions(state)
                    maxQValue = 0
                    first = True
                    for action in possibleActions:
                        qValue = self.computeQValueFromValues(state, action)
                        if qValue > maxQValue or first:
                            maxQValue = qValue
                            first = False
                    if not first:
                        tempValues[state] = maxQValue
            self.values = tempValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for tStateAndProb in transitionStatesAndProbs:
            nextState = tStateAndProb[0]
            prob = tStateAndProb[1]
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            nextStateValue = self.values[nextState]
            qValue = qValue + (prob * (reward + discount * nextStateValue))

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None

        if self.mdp.isTerminal(state):
            return bestAction

        possibleActions = self.mdp.getPossibleActions(state)
        maxQValue = 0
        first = True
        for action in possibleActions:
            qValue = self.computeQValueFromValues(state, action)
            if qValue > maxQValue or first:
                maxQValue = qValue
                bestAction = action
                first = False

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numberOfStates = len(states)

        for iteration in range(self.iterations):
            state = states[iteration % numberOfStates]
            if self.mdp.isTerminal(state):
                continue
            else:
                possibleActions = self.mdp.getPossibleActions(state)
                maxQValue = 0
                first = True
                for action in possibleActions:
                    qValue = self.computeQValueFromValues(state, action)
                    if qValue > maxQValue or first:
                        maxQValue = qValue
                        first = False
                self.values[state] = maxQValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            else:
                for action in self.mdp.getPossibleActions(state):
                    for tStateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
                        nextState = tStateAndProb[0]
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state}

        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            else:
                value = self.values[state]
                maxQValue = 0
                first = True
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    if qValue > maxQValue or first:
                        maxQValue = qValue
                        first = False
                diff = abs(value - maxQValue)
                queue.update(state, -diff)

        for iteration in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            if self.mdp.isTerminal(s):
                continue
            else:
                maxQValue = 0
                first = True
                for action in self.mdp.getPossibleActions(s):
                    qValue = self.computeQValueFromValues(s, action)
                    if qValue > maxQValue or first:
                        maxQValue = qValue
                        first = False
                self.values[s] = maxQValue
            for p in predecessors[s]:
                if self.mdp.isTerminal(p):
                    continue
                else:
                    maxQValue = 0
                    first = True
                    for action in self.mdp.getPossibleActions(p):
                        qValue = self.computeQValueFromValues(p, action)
                        if qValue > maxQValue or first:
                            maxQValue = qValue
                            first = False
                    value = self.values[p]
                    diff = abs(value - maxQValue)
                    if diff > self.theta:
                        queue.update(p, -diff)

