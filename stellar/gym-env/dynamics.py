import numpy as np
#import cupy as np
import numba as nb
import pdb
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
"""
cw_discrete
chaser_discrete
cw_continous
chaser_continous
"""
class cw_discrete:

    def __init__(self):
        gravitational_constant = float(4041.804)
        a = float(42164000)
        n = np.sqrt(gravitational_constant / (a ** 3.0))

        self.A = np.array([[(4.0 - 3.0*np.cos(n)), 0, 0, ((1.0/n) * np.sin(n)), ((2.0/n)*(1-np.cos(n))), 0], 
                          [(6*(np.sin(n) - n)), 1, 0, ((2.0/n)*(np.cos(n) - 1)), ((1.0/n)*(4 * np.sin(n) - 3*n)), 0],
                          [0, 0, np.cos(n), 0, 0, ((1.0/n)*np.sin(n))],
                          [(3*n * np.sin(n)), 0, 0, np.cos(n), (2.0 * np.sin(n)), 0],
                          [((6*n) * (np.cos(n) - 1)), 0, 0, (-2 * np.sin(n)), (4 * np.cos(n) - 3), 0],
                          [0, 0, (-1 * (n) * np.sin(n)), 0, 0, np.cos(n)]], np.float64)

        self.B = np.array([[( (1 / n**2.0) * (1 - np.cos(n)) ), ( (1 / n**2.0) * ((2.0 * n) - (2.0 * np.sin(n))) ), 0],
                      [( (1.0 / n ** 2.0) * (2.0 * (np.sin(n) - n)) ), ((-3 / 2) + (4/(n ** 2.0)) * (1 - np.cos(n))), 0],
                      [0, 0, ((1.0/n**2.0) * (1.0 - np.cos(n)))],
                      [(np.sin(n) / n ), ((2.0 / n) * (1.0 - np.cos(n))), 0],
                      [((2.0 / n) * (np.cos(n) - 1.0)), (-3 + (4.0 / n) * np.sin(n)), 0],
                      [0, 0, (np.sin(n) / n)]], np.float64)

    def step(self, state, action, mass):
        """
        input:
             -state:(6,) array
             -action:(3,) array
             -mass: int or float
        """

        x = np.array(state, dtype=np.float64)
        u = np.array(action, dtype=np.float64)
        #u = u / mass

        x = np.reshape(x, (6,1))
        u = np.reshape(u, (3,1))
        #pdb.set_trace()
        #print(f'action used in newtons {u}')
        u = u / mass
        #print(f'action divided by mass {mass}: {u}')
        x_next = np.matmul(self.A,x) + np.matmul(self.B,u)
        #x_next = (np.dot(self.A,x)) + ( np.dot(self.B,u) / mass )
        x_next = np.reshape(x_next, (6,))
        return x_next

class chaser_discrete(cw_discrete):

    def __init__(self, use_vbar, use_rbar):
        super().__init__()
        self.state_trace = []
        self.mass = 500.0 #500kg
        self.current_step = 0
        #self.state = self.rand_state()
        assert type(use_vbar) == bool, 'Input use_vbar must be boolean'
        assert type(use_rbar) == bool, 'Input use_rbar must be boolean'
        self.use_rbar = use_rbar
        self.use_vbar = use_vbar

        self.state = self.init_state()
        print(f'x0 is {self.state}')
        self.docking_point = np.array([0, 60, 0])
        self.theta_cone = 60

        self.slowzone_d = 500 #500 meters
        self.phase3_d = 100 #100 meters

    def is_vbar(self, input):
        assert type(input) == bool, 'Input in is_vbar must be boolean'

        #if using vbar disable rbar
        if input == True:
            self.use_rbar = False
        self.use_vbar = input

    def is_rbar(self, input):
        assert type(input) == bool, 'Input in is_rbar must be boolean'

        #if using rbar disable vbar
        if input == True:
            self.use_vbar = False

        self.use_rbar = input

    def init_v_bar(self):
        """
        The starting position for the V-bar approach was 
        [0, 1000, 0] m ± [100, 100, 5] m

        reference : https://www.researchgate.net/profile/Richard-Linares/publication/331135519_Spacecraft_Rendezvous_Guidance_in_Cluttered_Environments_via_Reinforcement_Learning/links/5c672585a6fdcc404eb44d45/Spacecraft-Rendezvous-Guidance-in-Cluttered-Environments-via-Reinforcement-Learning.pdf
        """

        v_bar_start = np.array([0,1000,0], np.float64)
        v_bar_range = np.array([100, 100, 5], np.float64)

        """
        two uniform distributions from [0,1) subtracted from each other
        to generate random floats between [-1,1]
        """
        percentage_offsets = np.random.random_sample(3) - np.random.random_sample(3)

        #offset from v_bar_start in units
        offsets = np.multiply(v_bar_range, percentage_offsets)

        #add offsets to v_bar_start
        pos = np.add(v_bar_start, offsets)
        #vel = np.random.randint(low=-10, high=10, size=3)
        range = np.random.randint(low=-2, high=2)
        vel = range * np.random.random_sample((3,))
        x0 = np.concatenate((pos, vel), axis=None, dtype=np.float64)
        return x0

    def init_r_bar(self):
        """
        The starting position for the V-bar approach was 
        [1000, 0, 0] m ± [100, 100, 5] m

        reference : https://www.researchgate.net/profile/Richard-Linares/publication/331135519_Spacecraft_Rendezvous_Guidance_in_Cluttered_Environments_via_Reinforcement_Learning/links/5c672585a6fdcc404>
        """

        r_bar_start = np.array([1000,0,0], np.float64)
        r_bar_range = np.array([100, 100, 5], np.float64)

        """
        two uniform distributions from [0,1) subtracted from each other
        to generate random floats between [-1,1]
        """
        percentage_offsets = np.random.random_sample(3) - np.random.random_sample(3)

        #offset from v_bar_start in units
        offsets = np.multiply(r_bar_range, percentage_offsets)

        #add offsets to v_bar_start
        pos = np.add(r_bar_start, offsets)
        vel = np.random.randint(low=-10, high=10, size=3)
        x0 = np.concatenate((pos, vel), axis=None, dtype=np.float64)

        return x0


    def rand_state(self):
        #revise
        pos = np.random.randint(low=-1000, high=1000, size=3)
        vel = np.random.randint(low=-10, high=10, size=3)

        state = np.concatenate((pos, vel), axis=None, dtype=np.float64)
        print("generated state")
        print(state)

        return state

    def init_state(self):

        """
        generates an x0 depending on generation settings (self.use_vbar, self.use_rbar)

        returns : np.ndarray size 3 dtype = np.float64
        """

        """
        demorgans used for clarification
        (notvbar and notrbar)
        not(vbar or rbar)
        """
        if not (self.use_vbar or self.use_rbar):
            print('random initalization')
            return self.rand_state()
        if self.use_vbar:
            print('using vbar')
            return self.init_v_bar()
        print('using rbar')
        return self.init_r_bar()

    def get_next(self, action):
        next_x = super().step(self.state, action, self.mass)
        return next_x

    def update_state(self, state):
        self.state_trace.append(state)
        self.state = state
        self.current_step += 1
        #print(f"state {self.state}, step {self.current_step}")

    def get_state_trace(self):
        return self.state_trace

    def reset(self):
        self.state_trace = []
        self.mass = 500.0 #500kg
        self.current_step = 0
        self.state = self.init_state()
        print(f'x0 is {self.state}')
        print("reset to default")
        print('-------------------')


class cw_continous:

    def __init__(self):
        gravitational_constant = float(3.986e+14)
        print(f'gravitational constant is {gravitational_constant}')
        a = float(42164000)
        n = np.sqrt(gravitational_constant / (a ** 3.0))
        m_chaser = 500.0 #kg


        self.A = np.array([[0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [(3*n**2), 0, 0, 0, 2*n, 0],
                     [0, 0, 0, (-1*2*n), 0, 0],
                     [0, 0, (-1*n**2), 0, 0, 0]], np.float64)

        self.B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], np.float64)


    def step(self, state, action, mass):
        """
        input:
             -state:(6,) array
             -action:(3,) array
             -mass: int or float
        """

        x = np.array(state, dtype=np.float64)
        u = np.array(action, dtype=np.float64)
        #u = u / mass
        #print(x)
        #print(f'x shape {x.shape}')
        x = np.reshape(x, (6))
        u = np.reshape(u, (3))
        #pdb.set_trace()
        #print(f'action used in newtons {u}')
        u = u / mass

        #print(f'control input divided by mass {u}')
        #dt = 0.05

        T = 1.0
        #n_slices = np.floor(T / dt).astype(np.int32)

        #time_slices = np.linspace(0, T, n_slices)
        #time_slices = np.floor(time_slices).astype(np.int32)

        #print(time_slices)
        #xB = np.zeros((6, n_slices))
        #xB[:, 0] = x

        A = self.A
        """
        for k in range(n_slices - 1):
            xB[:, k + 1] = np.linalg.pinv(np.eye(6) - A * dt) @ xB[:, k]
        """
        def ode_f(t, x, B, u):
            #print(f'x in ode {x.shape}')
            return A @ x + B @ u

        #vode_f = np.vectorize(ode_f)

        sol = solve_ivp(ode_f, (0, T), x, args=(self.B, u), method='RK45', vectorized=False)
        #print('solved')
        xRK4 = sol.y
        x_next = np.zeros(6, dtype=np.float64)

        """
        for idx, x in enumerate(xRK4):
            #print('x', x)
            x_next[idx] = x[-1]
        """

        x_next = xRK4[:, -1]
        #print(f'action divided by mass {mass}: {u}')
        #x_next = np.matmul(self.A,x) + np.matmul(self.B,u)
        #x_next = (np.dot(self.A,x)) + ( np.dot(self.B,u) / mass )
        #print('attempting to reshape')
        #x_next = np.reshape(x_next, (6,))
        assert x_next.shape == (6,), 'shape error'
        return x_next


class chaser_continous(cw_continous):
    def __init__(self, use_vbar, use_rbar):
        super().__init__()
        self.state_trace = []
        self.u_trace = []
        self.mass = 500.0 #500kg
        self.current_step = 0
        #self.state = self.rand_state()
        assert type(use_vbar) == bool, 'Input use_vbar must be boolean'
        assert type(use_rbar) == bool, 'Input use_rbar must be boolean'
        self.use_rbar = use_rbar
        self.use_vbar = use_vbar

        self.state = self.init_state()
        self.actuation = np.array([0, 0, 0], dtype=np.float64)
        print(f'x0 is {self.state}')
        self.docking_point = np.array([0, 60, 0])
        self.theta_cone = 60

        self.slowzone_d = 500 #500 meters
        self.phase3_d = 100 #100 meters

    def is_vbar(self, input):
        assert type(input) == bool, 'Input in is_vbar must be boolean'

        #if using vbar disable rbar
        if input == True:
            self.use_rbar = False
        self.use_vbar = input

    def is_rbar(self, input):
        assert type(input) == bool, 'Input in is_rbar must be boolean'

        #if using rbar disable vbar
        if input == True:
            self.use_vbar = False

        self.use_rbar = input

    def init_v_bar(self):
        """
        The starting position for the V-bar approach was 
        [0, 1000, 0] m ± [100, 100, 5] m

        reference : https://www.researchgate.net/profile/Richard-Linares/publication/331135519_Spacecraft_Rendezvous_Guidance_in_Cluttered_Environments_via_Reinforcement_Learning/links/5c672585a6fdcc404eb44d45/Spacecraft-Rendezvous-Guidance-in-Cluttered-Environments-via-Reinforcement-Learning.pdf
        """

        v_bar_start = np.array([0,800,0], np.float64)
        v_bar_range = np.array([200*2, 150*2, 200*2], np.float64)

        """
        two uniform distributions from [0,1) subtracted from each other
        to generate random floats between [-1,1]
        """
        percentage_offsets = np.random.random_sample(3) - np.random.random_sample(3)

        #offset from v_bar_start in units
        offsets = np.multiply(v_bar_range, percentage_offsets)

        #add offsets to v_bar_start
        pos = np.add(v_bar_start, offsets)
        #vel = np.random.randint(low=-10, high=10, size=3)
        range = np.random.randint(low=-2, high=2)
        vel = range * np.random.random_sample((3,))
        x0 = np.concatenate((pos, vel), axis=None, dtype=np.float64)
        return x0

    def init_r_bar(self):
        """
        The starting position for the V-bar approach was 
        [1000, 0, 0] m ± [100, 100, 5] m

        reference : https://www.researchgate.net/profile/Richard-Linares/publication/331135519_Spacecraft_Rendezvous_Guidance_in_Cluttered_Environments_via_Reinforcement_Learning/links/5c672585a6fdcc404>
        """

        r_bar_start = np.array([1000,0,0], np.float64)
        r_bar_range = np.array([100, 100, 5], np.float64)

        """
        two uniform distributions from [0,1) subtracted from each other
        to generate random floats between [-1,1]
        """
        percentage_offsets = np.random.random_sample(3) - np.random.random_sample(3)

        #offset from v_bar_start in units
        offsets = np.multiply(r_bar_range, percentage_offsets)

        #add offsets to v_bar_start
        pos = np.add(r_bar_start, offsets)
        vel = np.random.randint(low=-10, high=10, size=3)
        x0 = np.concatenate((pos, vel), axis=None, dtype=np.float64)

        return x0

    def rand_state(self):
        #revise
        pos = np.random.randint(low=-1000, high=1000, size=3)
        vel = np.random.randint(low=-10, high=10, size=3)

        state = np.concatenate((pos, vel), axis=None, dtype=np.float64)
        print("generated state")
        print(state)

        return state

    def init_state(self):

        """
        generates an x0 depending on generation settings (self.use_vbar, self.use_rbar)

        returns : np.ndarray size 3 dtype = np.float64
        """

        """
        demorgans used for clarification
        (notvbar and notrbar)
        not(vbar or rbar)
        """
        if not (self.use_vbar or self.use_rbar):
            print('random initalization')
            return self.rand_state()
        if self.use_vbar:
            print('using vbar')
            return self.init_v_bar()
        print('using rbar')
        return self.init_r_bar()

    def get_next(self, action):
        next_x = super().step(self.state, action, self.mass)
        return next_x

    def update_state(self, state):
        self.state_trace.append(state)
        self.state = state
        self.current_step += 1
        #print(f"state {self.state}, step {self.current_step}")

    def get_state_trace(self):
        return self.state_trace

    def update_u(self, actuation):
        #print(f'UPDATING U TRACE WITH {actuation} WITH TYPE {type(actuation)}')
        self.u_trace.append(actuation)
        self.actuation = actuation
        #self.state = state
        #self.current_step += 1
        #print(f"state {self.state}, step {self.current_step}")

    def get_u_trace(self):
        return self.u_trace


    def reset(self):
        self.state_trace = []
        self.u_trace = []
        self.mass = 500.0 #500kg
        self.current_step = 0
        self.state = self.init_state()
        self.actuation = np.array([0, 0, 0], dtype=np.float64)
        print(f'x0 is {self.state}')
        print("reset to default")
        print('-------------------')
