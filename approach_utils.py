from math import ceil, floor


# This fn rounds things to the nearest discrete step
# TODO: this could take a fn as argument instead of behavior str
def round_to_step(value, step_size, behavior='round'):
    if behavior == 'round':
        result = round(value / step_size) * step_size
    if behavior == 'floor':
        result = floor(value / step_size) * step_size
    if behavior == 'ceil':
        result = ceil(value / step_size) * step_size
    return result
