import math
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import seaborn as sns


def save_fig(name):
    plt.savefig(f"{name}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{name}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()


plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
mpl.rcParams['axes.linewidth'] = 0.2
plt.style.use('seaborn-white')

n_points = 100

# Mountain car 3D
fig = plt.figure(figsize=(6.5, 4))
ax = fig.add_subplot(projection='3d', computed_zorder=False)
ax.grid(False)
ax.view_init(elev=30, azim=-120)
x = np.linspace(-1.2, 0.6, n_points)
y = np.linspace(-1.2, 1.2, n_points)
x, y = np.meshgrid(x, y)
z = np.sin(3 * x) * 0.45 + 0.55

# Goal
goal_x = 0.48 * np.ones(n_points)
goal_y = np.linspace(-1.2, 1.2, n_points)
goal_z = 1.0 * np.ones(n_points)

# Symmetry
sym_x = -0.50 * np.ones(n_points)
sym_y = np.linspace(-1.2, 1.2, n_points)
sym_z = .85 * np.ones(n_points)

ax.plot_wireframe(x, y, z, color='#8da0cb', linewidth=1.0, alpha=0.6, zorder=1)
ax.scatter([-0.45], [0.0], [0.16], color="#ff7f00", s=100, zorder=2)
for i in range(12):
    ax.plot(goal_x + i / 100, goal_y, goal_z, color='#a6d854', linewidth=2, linestyle='-', zorder=3, alpha=0.3)
for i in range(5):
    ax.plot(sym_x + i/100, sym_y, sym_z, color='k', linewidth=2, linestyle=':', zorder=4, alpha=1)

# ax.set_title(f"Mountain Car 3D", y=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
save_fig(f"{plot_dir}/mountain_car_3D")

# Mountain car 3D Curved
fig = plt.figure(figsize=(6.5, 4))
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
ax.view_init(elev=30, azim=-120)
ax.grid(False)
domain = 1.2
x = np.linspace(-domain, domain, n_points)
y = np.linspace(-domain, domain, n_points)
x, y = np.meshgrid(x, y)
z = np.sin(3 * np.sqrt(x**2 + y**2) - math.pi / 2) * 0.45 + 0.55

# Goal
t = np.linspace(0., 2 * np.pi, n_points)


# Symmetry
sym_x = np.zeros(n_points)
sym_y = np.zeros(n_points)
sym_z = np.linspace(.3, 1., n_points)

ax.plot_wireframe(x, y, z, color='#8da0cb', linewidth=1.0, alpha=0.4, zorder=1)
ax.scatter([0.0], [0.0], [0.15], color="#ff7f00", s=100, zorder=0)
for i in range(12):
    goal_x = (1.0 + i / 100) * np.sin(t)
    goal_y = (1.0 + i / 100) * np.cos(t)
    goal_z = 1.0 * np.ones(n_points)
    ax.plot(goal_x, goal_y, goal_z, color='#a6d854', linewidth=2, linestyle='-', zorder=3, alpha=0.3)
ax.plot(sym_x, sym_y, sym_z, color='k', linewidth=2, linestyle='-.', zorder=0, alpha=1)
# ax.set_title(f"Mountain Car 3D Curved", y=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
save_fig(f"{plot_dir}/mountain_car_3D_curved")
