import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Parameters
h = 0.1
time = np.arange(0, 60, h)
m2 = 5
I2 = 10
l2 = 0.5
ics = (0, 0)

w2 = m2*9.8


def state_variables(x, _):
    return np.array([
        x[1],
        -2*l2/2*w2*np.cos(x[0])/(4*I2 + l2**2*m2)
    ])


t2 = np.zeros(time.size)
t2_dot = np.zeros(time.size)
t2[0], t2_dot[0] = ics

for i in range(time.size - 1):
    d = state_variables((t2[i], t2_dot[i]), None)
    t2[i + 1] = t2[i] + h*d[0]
    t2_dot[i + 1] = t2_dot[i] + h*d[1]

sol = odeint(state_variables, ics, time)

plt.title('Good Example of Numerical Error')
plt.plot(time, sol[:, 0], label='Odeint Solution')
plt.plot(time, t2, label="Euler's Method")
plt.show()

A = l2*np.exp(1j*np.array(sol[:, 0]))
Ax = np.real(A)
Ay = np.imag(A)

x_min, x_max = np.min(Ax), np.max(Ax)
y_min, y_max = np.min(Ay), np.max(Ay)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-l2 - 0.25, l2 + 0.25)
ax.set_ylim(-l2 - 0.25, l2 + 0.25)
ax.grid()


line = ax.plot([], [], color='maroon', marker='o')[0]
stamp = ax.text(0.05, 0.9, '', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white'))


def init():
    line.set_data([], [])
    stamp.set_text('')
    return [line, stamp]


def animate(index):
    line.set_data([0, Ax[index]], [0, Ay[index]])
    stamp.set_text(f'{time[index]:0.3f}')
    return [line, stamp]


# noinspection PyTypeChecker
ani = FuncAnimation(fig, animate, frames=range(time.size), interval=50, blit=True, init_func=init)
plt.show()

# ani.save('animations/single_pendulum.mp4', dpi=300)
