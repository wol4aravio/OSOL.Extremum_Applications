class Model_1:
    
    def __init__(self, control):
        self.control = control

    def __call__(self, t, y):
        x = y[0]
        u = self.control(t, x)
        return [u, u ** 2]

    def efficiency(self, sol):
        return 0.5 * sol.y[1][-1] + 0.5 * (sol.y[0][-1] ** 2), [self.control(t, x) for t, x in zip(sol.t, sol.y[0])]