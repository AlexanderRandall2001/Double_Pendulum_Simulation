import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DoublePendulum:
    """
    Create a double pendulum simulation using equations of motion derived from Lagrangian

    Args:
        theta1: Initial angle of upper pendulum from vertical (radians)
        theta2: Initial angle of lower pendulum from vertical (radians)
        l1: Length of upper link (m)
        l2: Length of lower link (m)
        m1: mass of upper link (kg)
        m2: mass of lower link (kg)
        g: Gravitational acceleration (m/s^2)
        dt: Time step for numerical integration (s)
    """

    def __init__(
            self,
            theta1: float,
            theta2: float,
            l1: float,
            l2: float,
            m1: float,
            m2: float,
            g: float,
            dt: float
    ):
        
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta1_dot = 0.0
        self.theta2_dot = 0.0
        self.theta1_ddot = None
        self.theta2_ddot = None
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.dt = dt
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.mass_matrix = None
        self.coriolis_vector = None
        self.gravity_vector = None
        self.trajectory = []

    def compute_MCG(self):
        c2 = np.cos(self.theta2)
        s2 = np.sin(self.theta2)
        c12 = np.cos(self.theta1 + self.theta2)

        self.mass_matrix = np.array([
            [self.m1*self.l1**2 + self.m2*(self.l1**2 + 2*self.l1*self.l2*c2 + self.l2**2),
            self.m2*(self.l1*self.l2*c2 + self.l2**2)],
            [self.m2*(self.l1*self.l2*c2 + self.l2**2),
            self.m2*self.l2**2]
        ])
        self.coriolis_vector = np.array([
            -self.m2*self.l1*self.l2*s2*(2*self.theta1_dot*self.theta2_dot + self.theta2_dot**2),
            self.m2*self.l1*self.l2*self.theta1_dot**2*s2
        ])
        self.gravity_vector = np.array([
            (self.m1 + self.m2)*self.l1*self.g*np.cos(self.theta1) + self.m2*self.g*self.l2*c12,
            self.m2*self.g*self.l2*c12
        ])

    def update_theta_ddot(self):
        theta_ddots = np.linalg.solve(self.mass_matrix, -self.coriolis_vector - self.gravity_vector)
        self.theta1_ddot = theta_ddots[0]
        self.theta2_ddot = theta_ddots[1]

    def update_theta_dot(self):
        self.theta1_dot = self.theta1_dot + self.theta1_ddot*self.dt
        self.theta2_dot = self.theta2_dot + self.theta2_ddot*self.dt

    def update_theta(self):
        self.theta1 = self.theta1 + self.theta1_dot*self.dt
        self.theta2 = self.theta2 + self.theta2_dot*self.dt
    
    def update_position(self):
        self.x1 = self.l1*np.sin(self.theta1)
        self.y1 = -self.l1*np.cos(self.theta1)
        self.x2 = self.l1*np.sin(self.theta1) + self.l2*np.sin(self.theta1 + self.theta2)
        self.y2 = -self.l1*np.cos(self.theta1) - self.l2*np.cos(self.theta1 + self.theta2)
    
    def swing(self):
        for _ in range(int(5*1/self.dt)):
            self.compute_MCG()
            self.update_theta_ddot()
            self.update_theta_dot()
            self.update_theta()
            self.update_position()
            self.trajectory.append(((self.x1, self.y1), (self.x2, self.y2)))
    
    def animate(self):
        interval = 1000 * self.dt
        fig = plt.figure()
        axes = fig.add_subplot(111)

        axes.set_xlim(-(self.l1 + self.l2) * 1.1, (self.l1 + self.l2) * 1.1)
        axes.set_ylim(-(self.l1 + self.l2) * 1.1, (self.l1 + self.l2) * 1.1)

        line1, = axes.plot([], [], ".-", lw=4, color='#66FF66')
        line2, = axes.plot([], [], ".-", lw=4, color='#66FF66')

        def update(frame):
            (x1, y1), (x2, y2) = self.trajectory[frame]
            line1.set_data([0, y1], [0, x1])
            line2.set_data([y1, y2], [x1, x2])
            return line1, line2
        
        ani = FuncAnimation(fig, update, frames = len(self.trajectory), interval = interval, blit = False)

        plt.show()
    
pendulum = DoublePendulum(1.7, 1.7, 1, 1, 1, 1, 9.8, 0.05)
pendulum.swing()
pendulum.animate()
        
