import numpy as np
import matplotlib.pyplot as plt

from solve_x_t import PNexapansion_x
from solve_x_t import m1, m2
from solve_x_t import times, x_vals, eval_function, x0, E

#actually finding e^-im*psi
def imganinary_exponent(psi_x, m):
    y = 0
    y_big = np.longdouble(y)
    y_big = np.exp(psi_x * -1j * m)

    return y_big

delta = np.sqrt((1-(4*E.nu)))

#solving for l=m=2 mode
H_2_2 = PNexapansion_x(m1, m2)
H_0 = 1
H_1 = (-107/42) + ((55*E.nu)/42)
H_2 = 2*np.pi
H_3 = (-2173/1512) - ((1069*E.nu)/216) + ((2047*(E.nu**2))/1512)

#solving for l=2, m=1
H_2_1 = PNexapansion_x(m1, m2)
H_1_0 = 1
H_1_1 = (-17/28) + ((5*E.nu)/7)
H_1_2 = (-1j/2) + np.pi - (2j*np.log10(2))
H_1_3 = (-43/126) - ((509*E.nu)/126) + ((79 * (E.nu**2))/168)

#solving for l=2, m=0
H_2_0 = PNexapansion_x(m1, m2)
H_0_0 = 1
H_0_1 = -(4075/4032) - ((67*E.nu)/48)
H_0_2 = -(151877213/67060224) - ((123815*E.nu)/44352) + ((205*E.nu**2)/352)

#solving for l=3, m=1
H_3_1 = PNexapansion_x(m1, m2)
H_3_1_0 = 1
H_3_1_1 = (-8/3) - ((2*E.nu)/3)
H_3_1_2 = (-7j/5) + np.pi - (2j * np.log10(2))
H_3_1_3 = (607/198) - (136 * E.nu/99) - (247 * E.nu**2/198)

#solving for l=3, m=2
H_3_2 = PNexapansion_x(m1, m2)
H_3_2_0 = 1 - (3*E.nu)
H_3_2_1 = (-193/90) + (145*E.nu/18) - (73*E.nu**2/18)
H_3_2_2 = -3j + (2*np.pi) + ((66j/5) - (6*np.pi))*E.nu
H_3_2_3 = (-1451/3960) - (17387*E.nu/3969) + (5557*E.nu**2/220) - (5341*E.nu**3/1320)

#solving for l=3, m=3
H_3_3 = PNexapansion_x(m1, m2)
H_3_3_0 = 1
H_3_3_1 = -4 + (2*E.nu)
H_3_3_2 = -(21j/5) + (3*np.pi) - (6j*np.log10(2)) + (6j*np.log10(3))
H_3_3_3 = (123/110) - (1838*E.nu/165) + (887*E.nu**2/330)

#using 2-PN corrections
#setting up H_2_2 mode
a = 1
H_2_2.setPowers((a, 1+a, 1.5+a, 2+a))
H_2_2.setConstants((H_0, H_1, H_2, H_3), alpha=8*np.sqrt(np.pi/5)*E.nu)

#setting up H_2_1 mode
b = 1.5
H_2_1.setPowers((b, 1+b, 1.5+b, 2+b))
H_2_1.setConstants((H_1_0, H_1_1, H_1_2, H_1_3), alpha = ((8j/3)*np.sqrt(np.pi/5)*delta))

#setting up H_2_0 mode
c = 1
H_2_0.setPowers((c, 1+c, 2+c))
H_2_0.setConstants((H_0_0, H_0_1, H_0_2))

#setting up H_3_1
d = 1.5
H_3_1.setPowers((d, 1+d, 1.5+d, 2+d))
H_3_1.setConstants((H_3_1_0, H_3_1_1, H_3_1_2, H_3_1_3), alpha= (1j/3)*E.nu*delta*np.sqrt((2*np.pi)/35))

#setting up H_3_2
e = 2
H_3_2.setPowers((e, 1+e, 1.5+e, 2+e))
H_3_2.setConstants((H_3_2_0, H_3_2_1, H_3_2_2, H_3_2_3), alpha=(8/3)*E.nu*np.sqrt(np.pi/7))

#setting H_3_3
f = 1.5
H_3_3.setPowers((f, 1+f, 1.5+f, 2+f))
H_3_3.setConstants((H_3_3_0, H_3_3_1, H_3_3_2, H_3_3_3), alpha = (-3j*delta*E.nu*np.sqrt((6*np.pi)/7)))

#get H_2_2_x
H_2_2_X = H_2_2.get_Eq()
H_2_2_x = eval_function(H_2_2_X, x_vals)

#H_2_1_x
H_2_1_X = H_2_1.get_Eq()
H_2_1_x = eval_function(H_2_1_X, x_vals)

#H_2_0_x
H_2_0_X = H_2_0.get_Eq()
H_2_0_x = eval_function(H_2_0_X, x_vals)

#H_3_1_x
H_3_1_X = H_3_1.get_Eq()
H_3_1_x = eval_function(H_3_1_X, x_vals)

#H_3_2_x
H_3_2_X = H_3_2.get_Eq()
H_3_2_x = eval_function(H_3_2_X, x_vals)

#H_3_3_x
H_3_3_X = H_3_3.get_Eq()
H_3_3_x = eval_function(H_3_3_X, x_vals)

#e^im*psi, assume psi_0 = 0
PSI = PNexapansion_x(m1, m2)
a = -2.5 #factor of x^-(5/2)
PSI_0 = 1
PSI_1 = ((-55/384)-(3715/32256))*(-32)*PSI.nu
PSI_2 = -10*np.pi
PSI_3 = -32*PSI.nu*((-27145/32256)-(15293365/(32514048 * PSI.nu)) - (3085*PSI.nu/4608)) 

PSI.setPowers((a, 1+a, 1.5+a, 2+a))
PSI.setConstants((PSI_0, PSI_1, PSI_2, PSI_3), alpha= -1/(32*PSI.nu))

PSI_X = PSI.get_Eq()
PSI_x = eval_function(PSI_X, x_vals)

#spin weighted olving the spherical harmonic
phi = 0
theta = np.pi/2

e_PSI_x_3 = np.real(imganinary_exponent(PSI_x, 3))
e_PSI_x_2 = np.real(imganinary_exponent(PSI_x, 2))
e_PSI_x_1 = np.real(imganinary_exponent(PSI_x, 1))
e_PSI_x_0 = np.real(imganinary_exponent(PSI_x, 0))

h_strain_20 = 0
h_strain_20 = np.longdouble(h_strain_20)
h_strain_20 = e_PSI_x_0 * H_2_0_x

h_strain_21 = 0
h_strain_21 = np.longdouble(h_strain_21)
h_strain_21 = e_PSI_x_1 * H_2_1_x

h_strain_22 = 0
h_strain_22 = np.longdouble(h_strain_22)
h_strain_22 = e_PSI_x_2 * H_2_2_x

h_strain_31 = 0
h_strain_31 = np.longdouble(h_strain_31)
h_strain_31 = e_PSI_x_1 * H_3_1_x

h_strain_32 = 0
h_strain_32 = np.longdouble(h_strain_31)
h_strain_32 = e_PSI_x_2 * H_3_2_x

h_strain_33 = 0
h_strain_33 = np.longdouble(h_strain_33)
h_strain_33 = e_PSI_x_3 * H_3_3_x

strain_fin = h_strain_20 + h_strain_21 + h_strain_22 + h_strain_31 + h_strain_32 + h_strain_33

plt.xlabel(r"$Time, t$")
plt.ylabel(r"$h$, strain")
plt.plot(times, h_strain_22, label='PN Corrections')
plt.legend(title=f"$m_1 ={PSI.m1}, m_2 = {PSI.m2}, c = 1, G = 1, x_0 = {x0[0]}$, 2nd Order")
plt.show()

# import sxs
# import numpy as np
# import matplotlib.pyplot as plt

# from solve_x_t import E

# sim = sxs.load("SXS:BBH:1132")
# h = sim.h

# metadata = sim.metadata
# horizons = sim.horizons

# idx1 = np.argmin(abs(horizons.A.time - metadata.reference_time))
# # plt.plot(h.t, h.data[:,h.index(2,2)])

# split_idx = int(len(h.t) * 0.5)
# trial = np.vstack((h.t[:split_idx], np.real(h.data[:,h.index(2,2)][:split_idx])))
# # trial.reshape(2, -1)
# print(np.shape(trial))
# plt.plot(trial[0], trial[1], label='Numerical Relativity')

# split_idx = int(len(times) * 0.95)

# plt.plot(times[:split_idx], h_strain_22[:split_idx], label='PN Corrections')
# plt.legend(title=f"$m_1 ={PSI.m1}, m_2 = {PSI.m2}, c = 1, G = 1, x_0 = {x0[0]}$, 2nd Order")
# plt.show()

experiment = np.asarray((times, (h_strain_20+h_strain_21+h_strain_22+h_strain_33)))