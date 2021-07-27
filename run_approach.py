import matplotlib.pyplot as plt
import numpy as np

from Red_Light_Approach.approach import Approach
from Red_Light_Approach.state import State 


# Load test case parameters
json_path = 'tests/test_params.json'

approach = Approach()
approach.configure(json_path)
approach.build_adjacency_matrix()
approach.backward_pass()
approach.forward_pass()


# Set up layer of adj matrix to plot
M = approach.adj_matrix
ss_size = approach.state_space_shape

## Plot how many states are reachable
if False:
    reachable_sum = [M[x,v].sum() for x, v in product(range(ss_size[0]), range(ss_size[1]))]
    reachable_sum = np.reshape(reachable_sum, ss_size)
    i, j = approach.state_to_indices(s)
    reachable_sum[i,j] = 7
    plt.imshow(reachable_sum, cmap='gray')
    plt.colorbar()
    plt.show()

#Plot reachable states from one state
if False:
    M[i,j][i,j] = True
    plt.imshow(M[i,j])
    plt.show()