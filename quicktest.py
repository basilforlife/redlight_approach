import json
import matplotlib.pyplot as plt
import numpy as np

from Red_Light_Approach.approach import *


# Load test case parameters
f = open('tests/test_params.json')
test1 = json.load(f)
f.close


def approach():
    approach = Approach()
    approach.set_compute_params(x_step=test1['compute_params']['x_step'],
                                v_step=test1['compute_params']['v_step'],
                                t_step=test1['compute_params']['t_step'])
    approach.set_world_params(v_max=test1['world_params']['v_max'],
                              a_max=test1['world_params']['a_max'])
    approach.set_traffic_light_params(first_support=test1['traffic_light_params']['first_support'],
                                      last_support=test1['traffic_light_params']['last_support'])
    approach.set_init_conditions(x_start=test1['init_conditions']['x_start'],
                                 v_start=test1['init_conditions']['v_start'])
    approach.compute_state_space_shape()
    return approach


def init_state():
    return State(pos=test1['init_conditions']['x_start'],
                 vel=test1['init_conditions']['v_start'])

# Set up layer of adj matrix to plot
approach = approach()
approach.build_adjacency_matrix()
M = approach.adj_matrix
m = np.reshape(M[0], approach.state_space_shape)

# Plot how many states are reachable
#reachable_sum = [M[i].sum() for i in range(approach.state_space_flat_size)]
#reachable_sum = np.reshape(reachable_sum, approach.state_space_shape)
#plt.imshow(reachable_sum, cmap='gray')
#plt.colorbar()
#plt.show()

#Plot reachable states from one state
idx = 1126
state = approach.idx_to_state(idx)
print(f'state.x = {state.x}; state.v = {state.v}')
plt.imshow(M[idx].reshape(approach.state_space_shape))
plt.show()


# State test
s = init_state()
