import numpy as np

class StraightLineAttacker:
    def __init__(self, v=200.0, bounds=((-500,500),(-500,500),(50,150)), dt=0.1):
        self.v = v
        self.bounds = bounds
        self.dt = dt
        self.reset()

    def reset(self, seed=None):
        x0 = np.random.uniform(*self.bounds[0])
        y0 = np.random.uniform(*self.bounds[1])
        z0 = np.random.uniform(*self.bounds[2])
        self.position = np.array([x0,y0,z0],dtype=np.float32)
        theta = np.random.uniform(0,2*np.pi)
        phi   = np.random.uniform(-np.pi/6,np.pi/6)
        self.direction = np.array([
            np.cos(phi)*np.cos(theta),
            np.cos(phi)*np.sin(theta),
            np.sin(phi)
        ],dtype=np.float32)
        self.velocity = self.v * self.direction
        return self.position.copy(), self.velocity.copy()

    def step(self):
        self.position += self.velocity * self.dt
        return self.position.copy(), self.velocity.copy()