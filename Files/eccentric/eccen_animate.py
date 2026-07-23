import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from eccentricity_main import x_coords, y_coords, x_coords2, y_coords2, t, wave, y0, m1, m2

width_px = 3840
height_px = 2160
dpi = 200

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_px/dpi, height_px/dpi)) 
ax2.set_box_aspect(1)
ax2.set_box_aspect(0.5)

skip = 100  # every 30th point

x_anim = x_coords[::skip]
y_anim = y_coords[::skip]
x2_anim = x_coords2[::skip]
y2_anim = y_coords2[::skip]

strain_anim = np.real(wave[::skip])
t = t[::skip]

# Set axis limits
padding = 1.1
xmin = min(x_anim.min(), x2_anim.min()) * padding
xmax = max(x_anim.max(), x2_anim.max()) * padding
ymin = min(y_anim.min(), y2_anim.min()) * padding
ymax = max(y_anim.max(), y2_anim.max()) * padding

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_aspect("equal")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Two masses
mass1, = ax1.plot([], [], 'bo', markersize=8, label=f'Mass 1 = {m1}, $x_0 = {y0[0]}, \iota_0 = {y0[1]}$')
mass2, = ax1.plot([], [], 'ro', markersize=8, label=f'Mass 1 = {m2}')

# Optional: show trails
trail1, = ax1.plot([], [], 'b-', alpha=0.6)
trail2, = ax1.plot([], [], 'r-', alpha=0.6)

ax1.legend()

ax2.set_xlim(t[0], t[-1])
ax2.set_ylim(strain_anim.min(), strain_anim.max())

strain_line, = ax2.plot([], [], 'g-')
strain_point, = ax2.plot([], [], 'ro')

line, = ax2.plot([], [], lw=2)
point, = ax2.plot([], [], 'bo', markersize=6)

def init():
    mass1.set_data([], [])
    mass2.set_data([], [])

    trail1.set_data([], [])
    trail2.set_data([], [])

    strain_line.set_data([], [])
    strain_point.set_data([], [])

    return (
        mass1, mass2,
        trail1, trail2,
        strain_line, strain_point
    )

def update(frame):
    trail = 50
    # Orbit
    start = max(0, frame - trail)

    mass1.set_data([x_anim[frame]], [y_anim[frame]])
    mass2.set_data([x2_anim[frame]], [y2_anim[frame]])

    trail1.set_data(x_anim[start:frame+1], y_anim[start:frame+1])
    trail2.set_data(x2_anim[start:frame+1], y2_anim[start:frame+1])

    # Strain
    strain_line.set_data(t[:frame+1], strain_anim[:frame+1])
    strain_point.set_data([t[frame]], [strain_anim[frame]])

    return (
        mass1, mass2,
        trail1, trail2,
        strain_line, strain_point
    )

ani = FuncAnimation(
    fig,
    update,
    frames=len(t),
    init_func=init,
    interval=10,
    blit=True
)

plt.tight_layout()
#plt.show()


ani.save("eccen_orbit_120.mp4", writer="ffmpeg", fps=240)
