import json
import numpy as np

import pytest

from Red_Light_Approach.approach import *


# Load test case parameters
f = open('tests/test_params.json')
test1 = json.load(f)
f.close

@pytest.fixture
def uniform_dist():
    distribution = Distribution()
    distribution.uniform_dist(10,20,1)
    return distribution

@pytest.fixture
def uniform_dist_result():
    x = np.zeros(10)
    return np.concatenate([x, x + 0.1])

@pytest.fixture
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

@pytest.fixture
def state_space_shape():
    return tuple(test1['state_space_shape'])

@pytest.fixture
def t_eval():
    return test1['t_eval']

@pytest.fixture
def init_state():
    return State(pos=test1['init_conditions']['x_start'],
                 vel=test1['init_conditions']['v_start'])
     
