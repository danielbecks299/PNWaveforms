import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
print(FFMpegWriter.isAvailable())

from find_r_t import paths, x0
from find_h_t import times, h_strain_22, PSI

x, y, x2, y2 = paths

#resolution
width_px = 1920
height_px = 1080
dpi = 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_px/dpi, height_px/dpi)) 
ax2.set_box_aspect(1)
ax2.set_box_aspect(0.5)

skip = 30  # every 30th point

x_anim = x[::skip]
y_anim = y[::skip]
x2_anim = x2[::skip]
y2_anim = y2[::skip]

strain_anim = np.real(h_strain_22[::skip])
t = times[::skip]

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
mass1, = ax1.plot([], [], 'bo', markersize=8, label=f'Mass 1 = {PSI.m1}, $x_0 = {x0}$')
mass2, = ax1.plot([], [], 'ro', markersize=8, label=f'Mass 1 = {PSI.m2}')

# Optional: show trails
trail1, = ax1.plot([], [], 'b-', alpha=0.6, lw=2)
trail2, = ax1.plot([], [], 'r-', alpha=0.6, lw=2)

ax1.legend()

ax2.set_xlim(t[0], t[-1])
ax2.set_ylim(strain_anim.min(), strain_anim.max())

strain_line, = ax2.plot([], [], 'g-')
strain_point, = ax2.plot([], [], 'go')

line, = ax2.plot([], [], lw=5)
point, = ax2.plot([], [], 'bo', markersize=10)

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
plt.show()

# ani.save("circ_binary_black_hole.gif", writer="pillow", fps=30)
#ani.save("circ_orbit_240.mp4", writer="ffmpeg", fps=240)
