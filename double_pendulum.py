import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Parameters
h = 0.05  # time step
time = np.arange(0, 60, h)  # Time interval
ics = (np.pi/2.01, np.pi/2.01, 0, 0)  # Initial conditions' theta_2, theta_3, theta_2_dot, theta_3_dot
m2, m3 = 5, 3  # in kilograms
I2, I3 = 1, 1  # in kg m^2
l2, l3 = 0.5, 0.25  # in m
T12 = lambda t: 0  # input torque

w2, w3 = 9.8*m2, 9.8*m3


def state_variables(x, t):
    t2, t3, t2_dot, t3_dot = x[0], x[1], x[2], x[3]
    denominator = 16*I2*I3 + 4*I2*l3**2*m3 + 4*I3*l2**2*m2 + 16*I3*l2**2*m3 + l2**2*l3**2*m2*m3 - 2*l2**2*l3**2*m3**2*np.cos(
        2*t2 - 2*t3) + 2*l2**2*l3**2*m3**2
    return np.array([
        t2_dot,
        t3_dot,
        (16*I3*T12(t) - 8*I3*t3_dot**2*l2*l3*m3*np.sin(t2 - t3) - 8*I3*l2*w2*np.cos(t2) - 16*I3*l2*w3*np.cos(
            t2) + 4*T12(t)*l3**2*m3 - 2*t2_dot**2*l2**2*l3**2*m3**2*np.sin(
            2*t2 - 2*t3) - 2*t3_dot**2*l2*l3**3*m3**2*np.sin(t2 - t3) - 2*l2*l3**2*m3*w2*np.cos(
            t2) - 2*l2*l3**2*m3*w3*np.cos(t2) + 2*l2*l3**2*m3*w3*np.cos(t2 - 2*t3))/denominator,
        (2*l3*(4*I2*t2_dot**2*l2*m3*np.sin(t2 - t3) - 4*I2*w3*np.cos(t3) - 4*T12(t)*l2*m3*np.cos(
            t2 - t3) + t2_dot**2*l2**3*m2*m3*np.sin(t2 - t3) + 4*t2_dot**2*l2**3*m3**2*np.sin(
            t2 - t3) + t3_dot**2*l2**2*l3*m3**2*np.sin(2*t2 - 2*t3) - l2**2*m2*w3*np.cos(t3) + l2**2*m3*w2*np.cos(
            t3) + l2**2*m3*w2*np.cos(2*t2 - t3) - 2*l2**2*m3*w3*np.cos(t3) + 2*l2**2*m3*w3*np.cos(
            2*t2 - t3)))/denominator
    ])


sol = odeint(state_variables, ics, time)
theta_2, theta_3 = np.array(sol[:, 0]), np.array(sol[:, 1])

A_points = l2*np.exp(1j*theta_2)
B_points = l3*np.exp(1j*theta_3) + A_points
Ax = np.real(A_points)
Ay = np.imag(A_points)
Bx = np.real(B_points)
By = np.imag(B_points)

cushion = 0.1
s = l2 + l3

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-s - cushion, s + cushion)
ax.set_ylim(-s - cushion, s + cushion)
ax.grid()

links = [ax.plot([], [], color='maroon', marker='o')[0], ax.plot([], [], color='deepskyblue', marker='o')[0],
         ax.text(0.05, 0.9, '', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white')),
         ax.plot([], [], color='black', marker='o', ms=2)[0]]

f = 5  # The size of the tail


def init():
    links[0].set_data([], [])
    links[1].set_data([], [])
    links[2].set_text('')
    links[3].set_data([], [])
    return links


def animate(index):
    links[0].set_data([0, Ax[index]], [0, Ay[index]])
    links[1].set_data([Ax[index], Bx[index]], [Ay[index], By[index]])
    links[2].set_text(f'{time[index]:.3f}')

    if index < f:
        links[3].set_data(Bx[:index], By[:index])
    else:
        links[3].set_data(Bx[index-f:index], By[index-f:index])

    return links


# noinspection PyTypeChecker
ani = FuncAnimation(fig, animate, frames=range(time.size), interval=50, blit=True, init_func=init)
plt.show()

# ani.save('animations/double_pendulum.mp4', dpi=300)
