import numpy as np
import matplotlib.pyplot as plt

from solve_x_t import PNexapansion_x
from solve_x_t import m1, m2, r
from solve_x_t import x_vals, eval_function, E, E_x

from find_h_t import PSI_x

E_binding = eval_function(E.get_Eq(), x_vals)

#find r, K = -binding_energy, K = -Gm1m2/r
def find_r_using_K(E):
    r = 0
    r_big = np.longdouble(r)
    K_energy = -(0.5)*(eval_function(E_x, x_vals))

    r_big = (-1/(2*K_energy)) * E.G * E.m1 * E.m2
    return r_big

#using gamma PN-parameters
Gamma = PNexapansion_x(m1, m2, r)
Gamma.setPowers(3)

G_0 = 1
G_1 = 1 - (Gamma.nu/3)
G_2 = 1 - (65*Gamma.nu/12)

Gamma.setConstants((0, G_0, G_1, G_2))
gamma = eval_function(Gamma.get_Eq(), x_vals)
r_gamma = (Gamma.G * Gamma.M)/(Gamma.c**2 * gamma) #the separation between the 2 bhs

#recall, m1r1 + m2r2 = 0
r1 = (Gamma.m2/Gamma.M)*r_gamma
r2 = (Gamma.m1/Gamma.M)*r_gamma


#converting to cartesian coordinates using kinetic energy
#combined_r = find_r_using_K(E)

#x = combined_r * np.cos(PSI_x)
#y = combined_r * np.sin(PSI_x)

#x2 = -combined_r * np.cos(PSI_x)
#y2 = -combined_r * np.sin(PSI_x)

#converting to cartesian using Gamma expansion
x = r1 * np.cos(PSI_x)
y = r1 * np.sin(PSI_x)

x2 = -r2 * np.cos(PSI_x)
y2 = -r2 * np.sin(PSI_x)

fig, ax = plt.subplots()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.plot(x, y, label=f'Mass 1 = {Gamma.m1}, $x_0 = {x_vals[0]}$')
plt.plot(x2, y2, label=f'Mass 2 = {Gamma.m2}')
plt.legend()
plt.show()


