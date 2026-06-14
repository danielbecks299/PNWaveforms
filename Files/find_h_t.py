import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from solve_x_t import PNexapansion_x
from solve_x_t import m1, m2, r
from solve_x_t import times, x_vals, eval_function

#solving for l=m=2 mode
H_2_2 = PNexapansion_x(m1, m2, r)
H_0 = 1
H_1 = (-107/42) + ((55*H_2_2.nu)/42)
H_2 = 2*np.pi
H_3 = (-2173/1512)-((1069*H_2_2.nu)/216)+((2047*(H_2_2.nu**2))/1512)

#solving for l=2, m=1
H_2_1 = PNexapansion_x(m1, m2, r)
H_1_0 = 1
H_1_1 = (-17/28) + ((5*H_2_1.nu)/7)
H_1_2 = (-1j/2) + np.pi - (2j*np.log(2))
H_1_3 = (-43/126) - ((509*H_2_1.nu)/126) + ((79 * (H_2_1.nu**2))/168)

#only working with 2-PN corrections
a = 1
H_2_2.setPowers((0+a, 1+a, 1.5+a, 2+a))
H_2_2.setConstants((H_0, H_1, H_2, H_3), alpha=8*np.sqrt(np.pi/5)*H_2_2.nu)

#setting up H_2_1 mode
b = 1.5
delta = np.sqrt(H_2_1.nu*(1-(4*H_2_1.nu)))
H_2_1.setPowers((0+b, 1+b, 1.5+b, 2+b))
H_2_1.setConstants((H_1_0, H_1_1, H_1_2, H_1_3), alpha = ((8j/3)*np.sqrt(np.pi/5)*delta))


#get H_2_2_x
H_2_2_X = H_2_2.get_Eq()
H_2_2_x = eval_function(H_2_2_X, x_vals)

#H_2_1_x
H_2_1_X = H_2_1.get_Eq()
H_2_1_x = eval_function(H_2_1_X, x_vals)

#e^im*psi, assume psi_0 = 0
PSI = PNexapansion_x(m1, m2, r)
a = 2.5 #factor of x^-(5/2)
PSI_0 = 1
PSI_1 = ((-55/384)-(3715/32256))*-32*PSI.nu
PSI_2 = -10*np.pi
PSI_3 = -32*PSI.nu*((-27145/32256)-(15293365/(32514048 * PSI.nu)) - (3085*PSI.nu/4608)) 

PSI.setPowers((0-a, 1-a, 1.5-a, 2-a))
PSI.setConstants((PSI_0, PSI_1, PSI_2, PSI_3), alpha= -1/(32*PSI.nu))

PSI_X = PSI.get_Eq()
PSI_x = eval_function(PSI_X, x_vals)

#actually finding e^im*psi
def imganinary_exponent(psi_x, m):
    y = 0
    y_big = np.longdouble(y)
    y_big = np.exp(psi_x * -1j * m)

    return y_big

e_PSI_x_2 = imganinary_exponent(PSI_x, 2)
e_PSI_x_1 = imganinary_exponent(PSI_x, 1)

h_strain_1 = 0
h_strain_1 = np.longdouble(h_strain_1)
h_strain_1 = e_PSI_x_1 * H_2_1_x

h_strain_2 = 0
h_strain_2 = np.longdouble(h_strain_2)
h_strain_2 = e_PSI_x_2 * H_2_2_x

strain_fin = np.real(h_strain_1 + h_strain_2)

plt.plot(times, strain_fin)
plt.show()
