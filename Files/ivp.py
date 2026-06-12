import sxs
import numpy as np
import matplotlib.pyplot as plt

from solve_x_t import E

sim = sxs.load("SXS:BBH:1132")

metadata = sim.metadata
horizons = sim.horizons

initial_orbital_frequency = np.linalg.norm(metadata.reference_orbital_frequency)
PN_x = ((initial_orbital_frequency)*E.M)**(2/3)

print(r"Initial orbital frequency omega is:", initial_orbital_frequency)
print(r"x = ", PN_x)

idx1 = np.argmin(abs(horizons.A.time - metadata.reference_time))
plt.plot(horizons.A.coord_center_inertial[idx1:,0], horizons.A.coord_center_inertial[idx1:,1])
plt.plot(horizons.B.coord_center_inertial[idx1:,0], horizons.B.coord_center_inertial[idx1:,1])
plt.xlabel('x position')
plt.ylabel('y position')

#plt.show()