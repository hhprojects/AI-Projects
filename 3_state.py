import numpy as np
import pandas as pd

#Define MDP Parameters
states = ["s0", "s1", "s2"]
actions = ["a0", "a1"]
gamma = 0.9 #Discount Factor
theta = 1e-4 #Convergence Threshold

#Transition probabilities and reward (s', P, R) for (s, a) -> s'
transition_rewards = {
    "s0": {
        "a0": [("s0", 0.5, 0), ("s2", 0.5, 0)],
        "a1": [("s2", 1, 0)]
    },
    "s1": {
        "a0": [("s0", 0.7, 5), ("s1", 0.1, 0), ("s2", 0.2, 0)],
        "a1": [("s1", 0.95, 0), ("s2", 0.05, 0)]
    },
    "s2": {
        "a0": [("s0", 0.4, 0), ("s1", 0.6, 0)],
        "a1": [("s0", 0.3, -1), ("s1", 0.3, 0), ("s2", 0.4, 0)]
    }
}

#Initialize value function
V = {s: 0 for s in states}

