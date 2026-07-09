import sxs
import numpy as np
import matplotlib.pyplot as plt

from solve_x_t import E

sim = sxs.load("SXS:BBH:1132")
h = sim.h

metadata = sim.metadata
horizons = sim.horizons

initial_orbital_frequency = np.linalg.norm(metadata.reference_orbital_frequency)

print(r"Initial orbital frequency omega is:", initial_orbital_frequency)

idx1 = np.argmin(abs(horizons.A.time - metadata.reference_time))
#plt.plot(h.t, h.data[:,h.index(2,2)])

split_idx = int(len(h.t) * 0.8)
trial = np.vstack((h.t[:split_idx], np.real(h.data[:,h.index(2,2)][:split_idx])))
# trial.reshape(2, -1)
# print(np.shape(trial))
# plt.plot(trial[0], trial[1])

# plt.plot(horizons.A.coord_center_inertial[idx1:,0], horizons.A.coord_center_inertial[idx1:,1])
# plt.plot(horizons.B.coord_center_inertial[idx1:,0], horizons.B.coord_center_inertial[idx1:,1])
# plt.xlabel('x position')
# plt.ylabel('y position')

#plt.plot(h.t, h.data[:,h.index(2,2)])
plt.show()