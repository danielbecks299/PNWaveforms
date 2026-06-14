import numpy as np
import matplotlib.pyplot as plt

from solve_x_t import PNexapansion_x
from solve_x_t import m1, m2, r
from solve_x_t import times, x_vals, eval_function, E, E_x

from find_h_t import PSI_x

E_binding = eval_function(E.get_Eq(), x_vals)

#find r, K = -binding_energy
def find_r_using_K(E):
    r = 0
    r_big = np.longdouble(r)
    K_energy = -(eval_function(E_x, x_vals))

    r_big = (-1/(4*K_energy)) * E.G * E.m1 * E.m2
    return r_big

#combined_r = find_r_using_K(E)

#using gamma PN-parameters
Gamma = PNexapansion_x(m1, m2, r)
Gamma.setPowers(3)

G_0 = 1
G_1 = 1 - (Gamma.nu/3)
G_2 = 1 - (65*Gamma.nu/12)

Gamma.setConstants((0, G_0, G_1, G_2))
gamma = eval_function(Gamma.get_Eq(), x_vals)
r_gamma = (1/2)*(Gamma.G * Gamma.M)/(Gamma.c**2 * gamma)

#converting to cartesian coordinates using kinetic energy
#x = combined_r * np.cos(PSI_x)
#y = combined_r * np.sin(PSI_x)

#x2 = -combined_r * np.cos(PSI_x)
#y2 = -combined_r * np.sin(PSI_x)

#converting to cartesian using Gamma expansion
x = r_gamma * np.cos(PSI_x)
y = r_gamma * np.sin(PSI_x)

x2 = -r_gamma * np.cos(PSI_x)
y2 = -r_gamma * np.sin(PSI_x)

plt.plot(x, y)
plt.plot(x2, y2)
plt.show()


