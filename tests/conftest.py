import json

import pytest

from redlight_approach.approach import Approach
from redlight_approach.distribution import ArbitraryDistribution, UniformDistribution
from redlight_approach.state import State

# Load test case parameters
param_filename = "parameter_files/original.json"
with open(param_filename) as f:
    test1 = json.load(f)


@pytest.fixture
def uniform_dist():
    return UniformDistribution(10, 20, 1)


@pytest.fixture
def arbitrary_dist():
    return ArbitraryDistribution([0, 0, 0, 0.1, 0.1, 0, 0, 0.2, 0.3, 0.2, 0.1, 0], 0.5)


@pytest.fixture
def approach():
    return Approach(param_filename)


@pytest.fixture
def state_space_shape():
    return (101, 37)


@pytest.fixture
def t_eval():
    return 27.2


@pytest.fixture
def init_state():
    return State(
        pos=test1["init_conditions"]["x_start"], vel=test1["init_conditions"]["v_start"]
    )


@pytest.fixture
def ran_red_state():
    return State(1, 10)


@pytest.fixture
def will_run_red_state():
    return State(-1, 10)


@pytest.fixture
def state_bounds():
    return [-100, 0, 0, 18]
