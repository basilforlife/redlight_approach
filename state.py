from typing import Optional, Tuple

Bounds = Tuple[float, float, float, float]


class State:
    """Represents the state of a vehicle

    This class stores the state (including position and velocity) of a vehicle.
    It assumes a continuous state space that is optionally bounded
    """

    def __init__(self, pos: float, vel: float, bounds: Optional[Bounds] = None) -> None:
        """Initializes a state object and checks bounds if provided

        Parameters
        ----------
        pos
            The position of the vehicle [m]
        vel
            The velocity of the vehicle [m/s]
        bounds
            The state space boundaries, in the order (x_min, x_max, v_min, v_max) [m]

        Raises
        ------
        ValueError
            If the state is not within the provided boundaries
        """
        self.x = pos
        self.v = vel
        if bounds:
            self.check_bounds(bounds)

    def __repr__(self) -> str:
        """Returns a string with the state vector

        Returns
        -------
        str
            A string describing the state
        """
        return f"State object: x = {self.x}; v = {self.v}"

    def __eq__(self, other: "State") -> bool:
        """Implements __eq__ for State object

        Parameters
        ----------
        other
            Object to compare to State object

        Returns
        -------
        bool
            True if the state vector is equivalent, False otherwise
        """
        if isinstance(other, State):
            return other.x == self.x and other.v == self.v
        else:
            return False

    def check_bounds(self, bounds: Bounds) -> None:
        """Raises a ValueError if the state is not within the bounds

        Parameters
        ----------
        bounds
            The state space boundaries, in the order (x_min, x_max, v_min, v_max) [m]

        Raises
        ------
        ValueError
            If the state is not within the provided boundaries
        """
        if (
            self.x < bounds[0]
            or self.x > bounds[1]
            or self.v < bounds[2]
            or self.v > bounds[3]
        ):
            raise ValueError("State does not lie within state space")
