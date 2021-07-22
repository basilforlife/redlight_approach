# State class holds a vehicle state
class State:
    def __init__(self, pos, vel, bounds=None):
        self.x = pos
        self.v = vel
        if bounds:
            self.check_bounds(bounds)

    def __repr__(self):
        return f'State object: x = {self.x}; v = {self.v}'

    def __eq__(self, other):
        if isinstance(other, State):
            return other.x == self.x and other.v == self.v
        else:
            return False

    def check_bounds(self, bounds):
        if self.x < bounds[0] or \
           self.x > bounds[1] or \
           self.v < bounds[2] or \
           self.v > bounds[3]:
            raise IndexError('State does not lie within state space')
