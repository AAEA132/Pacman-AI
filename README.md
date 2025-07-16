# Pacman-AI: Intelligent Agents based on UC Berkeley CS 188

> Course: Principles & Applications of Artificial Intelligence – Amirkabir University of Technology

> Course content: [CS188 - Introduction to AI, UC Berkeley](https://inst.eecs.berkeley.edu/~cs188/fa22/)

> Semester: Fall 2022

This repository contains four AI projectst implemented as part of the **Principles & Applications of Artificial Intelligence** course at my university. Based on **UC Berkeley’s CS 188 "Pacman Projects"** they explore fundamental AI concepts — including **search algorithms**, **adversarial search**, **reinforcement learning**, and **probabilistic inference** — using agents that operate in uncertain grid-based environments.

Each part focuses on the implementation and design of intelligent agents that make decisions based on the environment, opponents, or hidden variables using different AI techniques.

## Features
The project is divided into four parts, each focusing on a different AI technique: **Search Algorithms**, **Adversarial Multiagent Search**, **Reinforcement Learning**, and **Probabilistic Inference**.

### Part 1: Search
Implemented fundamental search algorithms to help Pacman navigate mazes and solve pathfinding problems, such as reaching goal points, collecting all food, or visiting all maze corners — while accounting for search cost and path optimality. This includes both uninformed (DFS, BFS, UCS) and informed (A*) strategies, with custom heuristics designed for admissibility and consistency to reduce computation.

### Part 2: Multiagent Search
This part extended the single-agent search problems to multiagent ones. This requires reasoning not only about Pacman's moves as, but also the possible actions of opponents. Pacman has to act optimally in environments with adversarial or stochastic agents (Ghosts), using algorithms like Minimax, Alpha-Beta pruning for adversarial search and Expectimax for when ghosts are stochastic. A custom evaluation function is also developed to improve decision-making under partial observability.

### Part 3: Reinforcement Learning
This part explores reinforcement learning, where agents learn how to act through interaction with the environment and trial-and-error, instead of being told how to. Agents are trained using value iteration and Q-learning to learn optimal policies that maximize long-term rewards. Approximate Q-learning is also used to scale learning to larger state spaces by using features instead of explicit states. With sufficient exploration, the agents converge to strong, reward-maximizing policies.

### Part 4: GhostBusters (Probabilistic Inference)
In this part, Pacman uses probabilistic inference to track and hunt invisible ghosts with his noisy distance sensors. The problem is modeled using Bayesian Networks and Hidden Markov Models (HMMs). Through exact and approximate inference (Bayes Nets & HMMs), the agent estimates and updates ghost locations over time using observations and moves greedily toward the most likely ones.

## How to Run

Clone the repository:

    git clone https://github.com/AAEA132/Pacman-AI.git
    cd Pacman-AI

Each part includes an autograder for testing correctness:

    # Run all tests for a part in its directory
    python autograder.py
    
    # Example for testing a specific question (e.g., Question 1)
    python autograder.py -q q1

## Acknowledgements
This project is based on the UC Berkeley’s CS 188: Introduction to Artificial Intelligence course. Full credit to the original course staff and original authors for the Pacman AI framework. All original materials can be found at [cs188.berkeley.edu](https://inst.eecs.berkeley.edu/~cs188/fa22/).

## License
This project is licensed under the MIT License. You are free to use it for academic, personal, or research purposes.