import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.signal import find_peaks, correlate

from solve_x_t import PNexapansion_x, E, F, eval_function, pole_event, limit, t_span, step, x0
from find_h_t import PSI, imganinary_exponent, experiment

dummy_E = PNexapansion_x()
dummy_F = PNexapansion_x()
dummy_PSI = PNexapansion_x()
dummy_H21 = PNexapansion_x()
dummy_H22 = PNexapansion_x()
dummy_H33 = PNexapansion_x()

def f_min(parameters, target_strain=experiment, plot=False):
    #building the PNequations that will change with the minimization algorithm
    M = parameters[0]
    nu = parameters[1]
    x0 = [parameters[2]]

    E_0 = 1
    E_1 = (-3/4) - (nu/12)
    E_2 = (-27/8) + (19*nu/8) - ((nu**2)/24)

    dummy_E.setPowers(E.p)
    dummy_E.setConstants((0, E_0, E_1, E_2), alpha = -(0.5)*(dummy_E.c)**2*(M)*nu)
    dummy_dE_dx = dummy_E.differentiate()

    F_0 = 1
    F_1 = (-1247/336) - ((35*nu)/12)
    F_2 = 4*np.pi
    F_3 = -(44711/9072) + (9271*nu/504) + (65*(nu**2)/18)

    dummy_F.setPowers(F.p)
    dummy_F.setConstants((F_0, F_1, F_2, F_3), alpha = 32*(dummy_F.c**5)*(nu**2)/(5*dummy_F.G)) 

    dummy_F_x = dummy_F.get_Eq()

    #had to redefine the ODE equations in terms of these 'dummy' objects
    def ode(t, x):
        xx = float(x[0])
        y = -eval_function(dummy_F_x, xx)/eval_function(dummy_dE_dx, xx)

        return y

    sol = solve_ivp(ode, t_span, x0, events=pole_event, method='RK45', t_eval=np.linspace(0, limit, step)) 

    if (not sol.success) or sol.y.size == 0:
        return np.inf

    solution_dummy = sol.y[0] #just getting the x(t) values

    #solving the PN equations for strain
    PSI_0 = 1
    PSI_1 = ((-55/384)-(3715/32256))*(-32)*nu
    PSI_2 = -10*np.pi
    PSI_3 = -32*nu*((-27145/32256)-(15293365/(32514048 * nu)) - (3085*nu/4608)) 

    dummy_PSI.setPowers(PSI.p)
    dummy_PSI.setConstants((PSI_0, PSI_1, PSI_2, PSI_3), alpha= -1/(32*nu))

    dummy_PSI_X = dummy_PSI.get_Eq()
    dummy_PSI_x = eval_function(dummy_PSI_X, solution_dummy)

    dummy_e_PSI_x_1 = np.real(imganinary_exponent(dummy_PSI_x, 1))
    dummy_e_PSI_x_2 = np.real(imganinary_exponent(dummy_PSI_x, 2))
    dummy_e_PSI_x_3 = np.real(imganinary_exponent(dummy_PSI_x, 3))

    #solving for H_2,1
    Delta = nu*(1-(4*nu))

    if Delta <= 0:
        delta = 0
    else:
        delta = np.sqrt(Delta)

    b = 1
    H_1_0 = 1
    H_1_1 = (-17/28) + ((5*nu)/7)
    H_1_2 = (-1j/2) + np.pi - (2j*np.log10(2))
    H_1_3 = (-43/126) - ((509*nu)/126) + ((79 * (nu**2))/168)

    dummy_H21.setPowers((b, 1+b, 1.5+b, 2+b))
    dummy_H21.setConstants((H_1_0, H_1_1, H_1_2, H_1_3), alpha = ((8j/3)*np.sqrt(np.pi/5)*delta))
    dummy_H21_X = dummy_H21.get_Eq()
    dummy_H21_x = eval_function(dummy_H21_X, solution_dummy)

    dummy_h_strain_21 =np.real(dummy_e_PSI_x_1 * dummy_H21_x)

    #solving for H_2,2
    H_0 = 1
    H_1 = (-107/42) + ((55*nu)/42)
    H_2 = 2*np.pi
    H_3 = (-2173/1512) - ((1069*nu)/216) + ((2047*(nu**2))/1512)

    a = 1
    dummy_H22.setPowers((a, 1+a, 1.5+a, 2+a))
    dummy_H22.setConstants((H_0, H_1, H_2, H_3), alpha=8*np.sqrt(np.pi/5)*nu)
    dummy_H22_X = dummy_H22.get_Eq()
    dummy_H22_x = eval_function(dummy_H22_X, solution_dummy)

    dummy_h_strain_22 =np.real(dummy_e_PSI_x_2 * dummy_H22_x)

    #solving for H_3,3
    f = 1.5
    H_3_3_0 = 1
    H_3_3_1 = -4 + (2*nu)
    H_3_3_2 = -(21j/5) + (3*np.pi) - (6j*np.log10(2)) + (6j*np.log10(3))
    H_3_3_3 = (123/110) - (1838*nu/165) + (887*nu**2/330)

    dummy_H33.setPowers((f, 1+f, 1.5+f, 2+f))
    dummy_H33.setConstants((H_3_3_0, H_3_3_1, H_3_3_2, H_3_3_3), alpha = (-3j*delta*nu*np.sqrt((6*np.pi)/7)))
    dummy_H33_X = dummy_H33.get_Eq()
    dummy_H33_x = eval_function(dummy_H33_X, solution_dummy)

    dummy_h_strain_33 =np.real(dummy_e_PSI_x_3 * dummy_H33_x)

    #sum them
    dummy_h_strain = dummy_h_strain_21 + dummy_h_strain_22 + dummy_h_strain_33

    # Choose a common grid
    t_common = target_strain[0]

    # Interpolate both waveforms onto the common grid
    original_strain_interpolated = np.interp(t_common, target_strain[0], target_strain[1])
    solution_dummy_interpolated = np.interp(t_common, sol.t, dummy_h_strain)

    target = original_strain_interpolated.copy()
    trial = solution_dummy_interpolated.copy()

    #normalization prioritizes finding the best timing as amplitude biases are disregarded and recalculated in the difference_vector
    target /= np.linalg.norm(target)
    trial /= np.linalg.norm(trial)

    #find the best time shift
    corr = correlate(target, trial, mode="full")
    shift = np.argmax(corr) - (len(trial) - 1)

    dt = shift * (t_common[1] - t_common[0]) #the subtraction is to find the timestep
    t_shifted = t_common + dt

    solution_aligned = np.interp(t_common, t_shifted, solution_dummy_interpolated, left=0, right=0)

    target_norm = original_strain_interpolated
    difference_vector = original_strain_interpolated - solution_aligned

    if plot:
        plt.plot(t_common, original_strain_interpolated, label='target')
        plt.plot(t_common, solution_aligned, label='attempt')
        plt.legend()
        plt.show()

    return np.linalg.norm(difference_vector) / np.linalg.norm(target_norm)

initial = np.array([1.49977434, 0.22458572, 0.07467666])

# initial_simplex = np.array([
#     initial,
#     initial + [0.1, 0, 0],
#     initial + [0, 0.1, 0],
#     initial + [0, 0, 0.01],
# ])

# M_values = np.arange(0.85, 1.2, 0.01)
# f_values = [f_min(M_value, 0.25, x0, experiment, False) for M_value in M_values]
# print(f_values)
# plt.plot(M_values, f_values)
# plt.yscale('log')
# plt.show()

bnds = Bounds([0, 0, 0], [np.inf, 0.25, 0.2])

res = minimize(f_min, initial, method='Nelder-Mead', args=(experiment), bounds = bnds, options={'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 10000, 'maxiter': 10000, 'disp': True})
print(res.x, res.fun)
f_min(res.x, experiment, True)

# xs = np.linspace(0.03, 0.13, 50)

# vals = [f_min([1, 0.25, x]) for x in xs

# plt.plot(xs, vals)
# plt.grid()
# plt.show()

#Newton-CG
#Powell


