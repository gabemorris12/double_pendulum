import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
h = 0.05  # time step
time = np.arange(0, 20, h)  # Time interval
ics = (0, 0, 0, 0)  # Initial conditions theta_2, theta_3, theta_2_dot, theta_3_dot
m2, m3 = 5, 3  # in kilograms
I2, I3 = 1, 1  # in kg m^2
l2, l3 = 0.5, 0.25  # in m
T12 = lambda t: 0  # input torque

w2, w3 = 9.8*m2, 9.8*m3


def state_variables(x, t):
    t2, t3, t2_dot, t3_dot = x[0], x[1], x[2], x[3]
    denominator = 16*I2*I3 + 4*I2*l3**2*m3 + 4*I3*l2**2*m2 - 16*I3*l2**2*m3 + l2**2*l3**2*m2*m3 + 2*l2**2*l3**2*m3**2*np.cos(
        2*t2 - 2*t3) - 2*l2**2*l3**2*m3**2
    return np.array([
        t2_dot,
        t3_dot,
        (16*I3*T12(t) + 8*I3*t3_dot**2*l2*l3*m3*np.sin(t2 - t3) - 8*I3*l2*w2*np.cos(t2) + 16*I3*l2*w3*np.cos(
            t2) + 4*T12(
            t)*l3**2*m3 + 2*t2_dot**2*l2**2*l3**2*m3**2*np.sin(2*t2 - 2*t3) + 2*t3_dot**2*l2*l3**3*m3**2*np.sin(
            t2 - t3) - 2*l2*l3**2*m3*w2*np.cos(t2) + 2*l2*l3**2*m3*w3*np.cos(t2) - 2*l2*l3**2*m3*w3*np.cos(
            t2 - 2*t3))/denominator,
        2*l3*(4*I2*t2_dot**2*l2*m3*np.sin(t2 - t3) - 4*I2*w3*np.cos(t3) - 4*T12(t)*l2*m3*np.cos(
            t2 - t3) + t2_dot**2*l2**3*m2*m3*np.sin(t2 - t3) - 4*t2_dot**2*l2**3*m3**2*np.sin(
            t2 - t3) - t3_dot**2*l2**2*l3*m3**2*np.sin(2*t2 - 2*t3) - l2**2*m2*w3*np.cos(t3) + l2**2*m3*w2*np.cos(
            t3) + l2**2*m3*w2*np.cos(2*t2 - t3) + 2*l2**2*m3*w3*np.cos(t3) - 2*l2**2*m3*w3*np.cos(
            2*t2 - t3))/denominator
    ])


theta_2 = np.zeros(time.size)
theta_3 = np.zeros(time.size)
theta_2_dot = np.zeros(time.size)
theta_3_dot = np.zeros(time.size)
theta_2[0], theta_3[0], theta_2_dot[0], theta_3_dot[0] = ics

for i in range(time.size - 1):
    d = state_variables((theta_2[i], theta_3[i], theta_2_dot[i], theta_3_dot[i]), time[i])
    theta_2[i + 1] = theta_2[i] + d[0]*h
    theta_3[i + 1] = theta_3[i] + d[1]*h
    theta_2_dot[i + 1] = theta_2_dot[i] + d[2]*h
    theta_3_dot[i + 1] = theta_3_dot[i] + d[3]*h

A_points = l2*np.exp(1j*theta_2)
B_points = l3*np.exp(1j*theta_3) + A_points
Ax = np.real(A_points)
Ay = np.imag(A_points)
Bx = np.real(B_points)
By = np.imag(B_points)

x_min = np.amin([Ax, Bx])
x_max = np.amax([Ax, Bx])
y_min = np.amin([Ay, By])
y_max = np.amax([Ay, Ay])

cushion = 0.1

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(x_min - cushion, x_max + cushion)
ax.set_ylim(y_min - cushion, y_max + cushion)
ax.grid()

links = [ax.plot([], [], color='maroon', marker='o')[0], ax.plot([], [], color='deepskyblue', marker='o')[0]]


def init():
    links[0].set_data([], [])
    links[1].set_data([], [])
    return links


def animate(index):
    links[0].set_data([0, Ax[index]], [0, Ay[index]])
    links[1].set_data([Ax[index], Bx[index]], [Ay[index], By[index]])
    return links


# noinspection PyTypeChecker
ani = FuncAnimation(fig, animate, frames=range(time.size), interval=50, blit=True, init_func=init)
plt.show()
