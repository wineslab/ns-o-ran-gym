from gymnasium import spaces



# In the traffic steering use case, the action is a combination between 
n_ues = 2
n_gnbs = 7  # scenario one has always 7 gnbs 
n_actions_ue = 7 # each UE can connect to a gNB identified by ID (from 2 to 8), 0 is No Action
action_space = spaces.MultiDiscrete([n_actions_ue] * n_ues *  n_gnbs)
print(action_space)
a = action_space.sample()
print(a)
print(len(a))

action_space = spaces.Discrete(n_actions_ue * n_ues *  n_gnbs)
print(action_space)
a = action_space.sample()
print(a)

print()