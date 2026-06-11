import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from solve_x_t import PNexapansion_x
from solve_x_t import m1, m2, r
from solve_x_t import times, values, eval_function

#solving for l=m=2 mode
H_2_2 = PNexapansion_x(m1, m2, r)
H_1 = 1
H_2 = (-107/42) + ((55*H_2_2.nu)/42)
H_3 = 2*np.pi
H_4 = (-2173/1512)-((1069*H_2_2.nu)/216)+((2047*(H_2_2.nu**2))/1512)

#only working with 2-PN corrections
a = 1
H_2_2.setPowers((0+a, 1+a, 1.5+a, 2+a))
H_2_2.setConstants((H_1, H_2, H_3, H_4), alpha=8*np.sqrt(np.pi/5)*H_2_2.nu)

#get H_2_2_x
H_2_2_x = H_2_2.get_Eq()
H_2_2_t = eval_function(H_2_2_x, values)

#e^im*psi, assume psi_0 = 0
PSI = PNexapansion_x(m1, m2, r)
a = 2.5 #factor of x^-(5/2)
PSI_0 = 1
PSI_1 = ((-55/384)-(3715/32256))*-32*PSI.nu
PSI_2 = -10*np.pi
PSI_3 = -32*PSI.nu*((-27145/32256)-(15293365/(32514048 * PSI.nu)) - (3085*PSI.nu/4608)) 

PSI.setPowers((0-a, 1-a, 1.5-a, 2-a))
PSI.setConstants((PSI_0, PSI_1, PSI_2, PSI_3), alpha= -1/(32*PSI.nu))

PSI_x = PSI.get_Eq()
PSI_t = eval_function(PSI_x, values)

#actually finding e^im*psi
def imganinary_exponent(equation, psi_t):
    y = 0
    y_big = np.longdouble(y)
    y_big += np.exp((equation.m1+equation.m2) * psi_t * -1j)

    return y_big

e_PSI_t = imganinary_exponent(PSI, PSI_t)

bleh = e_PSI_t * H_2_2_t

plt.plot(values, bleh)
plt.show()