import pandas as pd

# Define MDP Parameters
states = ["s0", "s1", "s2"]
actions = ["a0", "a1"]
gamma = 0.9 #Discount Factor
epsilon = 1e-3 #Convergence Threshold

# Transition probabilities and reward (s', P, R) for (s, a) -> s'
transition_model = {
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

# Initialize value function
V = {s: 0 for s in states}

count = 0
# Perform value iteration
while True:
    delta = 0
    v = V.copy()
    
    print(f"""
          iter  {count}  |   V(s0) = {round(v[states[0]],2)}   V(s1) = {round(v[states[1]],2)}   V(s0) = {round(v[states[2]],2)}
          """)
    
    for s in states:
        v[s] = max(sum(p * (r + gamma * V[s_next]) for s_next, p, r in transition_model[s][a]) for a in actions) #Bellman Optimality Equation
        delta = max(delta, abs(v[s] - V[s]))
    
    V = v # Update value function with new values
    
    # Check for convergence
    if delta < epsilon:
        break
    
    count += 1
    
# Extract optimal policy
policy = {}
for s in states:
    best_action = None
    best_value = -float("inf")

    for a in actions:
        value = sum(p * (r + gamma * V[s_next]) for s_next, p, r in transition_model[s][a])
        if value > best_value:
            best_value = value
            best_action = a

    policy[s] = best_action
    
    
# DataFrame to display the Value Function and Optimal Policy
df_combined = pd.DataFrame({
    "State": list(V.keys()),
    "Value": list(V.values()),
    "Optimal Action": [policy[s] for s in V.keys()]
})

print(df_combined)
