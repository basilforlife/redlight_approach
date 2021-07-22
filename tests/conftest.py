import json
import numpy as np

import pytest

from Red_Light_Approach.approach import Approach
from Red_Light_Approach.distribution import Distribution
from Red_Light_Approach.state import State


# Load test case parameters
with open('tests/test_params.json') as f:
    test1 = json.load(f)

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
    approach = Approach('tests/test_params.json')
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

@pytest.fixture
def ran_red_state():
    return State(1, 10)
     
