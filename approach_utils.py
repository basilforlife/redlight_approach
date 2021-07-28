from math import ceil, floor


def round_to_step(value: float, step_size: float, behavior: str = "round") -> float:
    """Rounds a value to a multiple of `step_size`

    Parameters
    ----------
    value
        value to discretize
    step_size
        discrete unit to round to a multiple of
    behavior
        One of 'round', 'floor', and 'ceil'.

    Returns
    -------
    float
        The rounded value
    """
    int_scaled_value = value / step_size
    if behavior == "round":
        result = round(int_scaled_value) * step_size
    if behavior == "floor":
        result = floor(int_scaled_value) * step_size
    if behavior == "ceil":
        result = ceil(int_scaled_value) * step_size
    return result
