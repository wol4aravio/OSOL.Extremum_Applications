from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


class BaseModel(ABC):
    
    def __init__(self, t0, t1, optimal_control, control):
        self.t0 = t0
        self.t1 = t1
        if control is not None:
            self.control = control
        else:
            self.control = optimal_control
        self.terminal_events = list()
        
    def get_controls(self, sol):
        return [self.control(t, y) for t, y in zip(sol.t, sol.y.T)]
    
    def calc_solution(self, initial_state, time_grid=None):
        sol = solve_ivp(
            fun=self, 
            method="RK45",
            t_span=[self.t0, self.t1], 
            y0=(initial_state.tolist() + [0.0]),
            dense_output=True, 
            t_eval=time_grid,
            events=self.terminal_events)
        if len(sol.t_events) > 0:
            terminal_value = sol.sol(sol.t_events[0])
            sol.y = np.append(sol.y, terminal_value, axis=1)
        
        return sol
    
    @abstractmethod
    def __call__(self, t, y):
        pass
    
    @abstractmethod
    def I(self, sol):
        pass
    
    @abstractmethod
    def I_term(self, sol):
        pass
    
    def I_term_conditions(self, sol):
        y_term = sol.y[:, -1]
        t_term = sol.t[-1]
        
        I_term_value = 0.0
        for term_event in self.terminal_events:
            I_term_value += (self.penalty * term_event(t_term, y_term)) ** 2
        return I_term_value

class Model_1(BaseModel):
    
    def __init__(self, control=None):
        def optimal_control(t, y):
            x = y[0]
            return x / (t - 2)
        super().__init__(
            t0=0.0, 
            t1=1.0, 
            optimal_control=optimal_control, 
            control=control)
    
    def __call__(self, t, y):
        x = y[0]
        u = self.control(t, y)
        return [u, u ** 2]
    
    def I(self, sol):
        return 0.5 * sol.y[1][-1]
    
    def I_term(self, sol):
        return 0.5 * (sol.y[0][-1] ** 2)
        
class Model_2(BaseModel):
    
    def __init__(self, control=None):
        def optimal_control(t, y):
            x1 = y[0]
            x2 = y[1]
            part_1 = 12 + 4 * ((2 - t) ** 2) * (5 - t)
            part_2 = 6 * (2 - t) * (4 - t)
            part_3 = 12 * (3 - t) + ((2 - t) ** 3) * (6 - t)
            return -(part_1 * x1 + part_2 * x2) / part_3
        super().__init__(
            t0=0.0, 
            t1=2.0, 
            optimal_control=optimal_control, 
            control=control)
    
    def __call__(self, t, y):
        x1 = y[0]
        x2 = y[1]
        u = self.control(t, y)
        return [u, x1, u ** 2]
    
    def I(self, sol):
        return 0.5 * sol.y[2][-1]
    
    def I_term(self, sol):
        return 0.5 * (sol.y[0][-1] ** 2 + sol.y[1][-1] ** 2)

class Model_3(BaseModel):
    
    def __init__(self, control=None, eps=1e-5, penalty=1e7):
        self.eps = eps
        self.penalty = penalty
        def optimal_control(t, y):
            x = y[0]
            return -np.sign(x)
        super().__init__(
            t0=0.0, 
            t1=100.0, 
            optimal_control=optimal_control, 
            control=control)
        
        def terminal_event(t, y):
            x = y[0]
            if x > eps or x < -eps:
                return np.abs(x)
            else:
                return 0.0
        terminal_event.terminal = True
        self.terminal_events = [terminal_event]
        
    def __call__(self, t, y):
        x = y[0]
        u = self.control(t, y)
        return [u, 1]
    
    def I(self, sol):
        return sol.y[-1, -1]
    
    def I_term(self, sol):
        return self.I_term_conditions(sol)
        
class Model_4(BaseModel):
    
    def __init__(self, control=None, eps=1e-5, penalty=1e7):
        self.eps = eps
        self.penalty = penalty
        def optimal_control(t, y):
            x1 = y[0]
            x2 = y[1]
            if x1 > -0.5 * x2 * np.abs(x2):
                return -1
            elif x1 < -0.5 * x2 * np.abs(x2):
                return 1
            else:
                return -np.sign(x2)
        super().__init__(
            t0=0.0, 
            t1=100.0, 
            optimal_control=optimal_control, 
            control=control)
        
        def terminal_event(t, y):
            x1 = y[0]
            x2 = y[1]
            result = 0.0

            if x1 > eps or x1 < -eps:
                result += np.abs(x1)
            if x2 > eps or x2 < -eps:
                result += np.abs(x2)

            return result
        terminal_event.terminal = True
        self.terminal_events = [terminal_event]
        
    def __call__(self, t, y):
        x1 = y[0]
        x2 = y[1]
        u = self.control(t, y)
        return [x2, u, 1]
    
    def I(self, sol):
        return sol.y[-1, -1]
    
    def I_term(self, sol):
        return self.I_term_conditions(sol)
