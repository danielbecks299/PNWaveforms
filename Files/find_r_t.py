import numpy as np
import matplotlib.pyplot as plt

from solve_x_t import PNexapansion_x
from solve_x_t import m1, m2, r
from solve_x_t import times, x_vals, eval_function, E

from find_h_t import PSI_x

E_binding = eval_function(E.get_Eq(), x_vals)

#find r
def find_r(k, K):
    r = 0
    r_big = np.longdouble(r)

    r_big = (-1/(2*k)) * E.G * E.m1 * E.m2
    return r_big

#kinetic energy is equal to the negative of the total conserved energy
K = PNexapansion_x(m1, m2, r)

K_0 = 1
K_1 = (3/2) + (K.nu/4)
K_2 = (27/8) + ((19*K.nu)/8) + ((K.nu**2)/24)

#2nd order kinetic energy
K.setPowers(3)
K.setConstants((0, K_0, K_1, K_2), alpha = ((-1/2)*(K.c**2)*(K.M)*(K.nu)))
K_x = K.get_Eq()

K_energy = eval_function(K_x, x_vals)
combined_r = find_r(K_energy, K)

#converting to cartesian coordinates
x = combined_r * np.cos(PSI_x)
y = combined_r * np.sin(PSI_x)

x2 = -combined_r * np.cos(PSI_x)
y2 = -combined_r * np.sin(PSI_x)

print(x)

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.plot(x2, y2)
#plt.show()

