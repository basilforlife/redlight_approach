import json

import pytest

from Red_Light_Approach.approach import Approach
from Red_Light_Approach.distribution import UniformDistribution
from Red_Light_Approach.state import State

# Load test case parameters
with open("tests/test_params.json") as f:
    test1 = json.load(f)


@pytest.fixture
def uniform_dist():
    return UniformDistribution(10, 20, 1)


@pytest.fixture
def approach():
    approach = Approach("tests/test_params.json")
    return approach


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
def state_bounds():
    return [-100, 0, 0, 18]
