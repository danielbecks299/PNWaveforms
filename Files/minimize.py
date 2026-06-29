import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize

from solve_x_t import PNexapansion_x, E, F, eval_function, pole_event, limit, t_span, step, E_0, E_1, E_2, F_0, F_1, F_2, F_3, x0
from find_h_t import H_2_2_X, PSI_X, imganinary_exponent, experiment

def f_min(parameters, x0=x0, target_strain=experiment, plot=False):
    #building the PNequations that will change with the minimization algorithm
    M, nu = parameters

    dummy_E = PNexapansion_x()
    dummy_F = PNexapansion_x()

    dummy_E.setM(M)
    dummy_E.setNu(nu)
    dummy_E.setPowers(E.p)
    dummy_E.setConstants((0, E_0, E_1, E_2), alpha = -(0.5)*(dummy_E.c)**2*(M)*nu)
    dummy_dE_dx = dummy_E.differentiate()

    dummy_F.setM(M)
    dummy_F.setNu(nu)
    dummy_F.setPowers(F.p)
    dummy_F.setConstants((F_0, F_1, F_2, F_3), alpha = 32*(dummy_F.c**5)*(nu**2)/(5*dummy_F.G)) 

    dummy_F_x = dummy_F.get_Eq()

    #had to redefine the ODE equations in terms of these 'dummy' objects
    def ode(t, x):
        xx = float(x[0])
        y = -eval_function(dummy_F_x, xx)/eval_function(dummy_dE_dx, xx)

        return y

    sol = solve_ivp(ode, t_span, x0, events=pole_event, method='RK45', t_eval=np.linspace(0, limit, step)) #where are these terms coming from....
    solution_dummy = sol.y[0] #just getting the x(t) values

    if not sol.success:
        return np.inf

    #solving the PN equations for strain
    solution_dummy_h22 = eval_function(H_2_2_X, solution_dummy)
    solution_dummy_psi_x = eval_function(PSI_X, solution_dummy)
    e_PSI_x_2 = np.real(imganinary_exponent(solution_dummy_psi_x, 2))
    dummy_h_strain =np.real(e_PSI_x_2 * solution_dummy_h22)

    #interpolation so that each trial and the final output match

    # Find overlapping time interval
    t_min = max(target_strain[0].min(), sol.t.min())
    t_max = min(target_strain[0].max(), sol.t.max())

    if t_max <= t_min:
        return np.inf

    # Choose a common grid
    N = max(len(target_strain[0]), len(sol.t))
    t_common = target_strain[0]

    # Interpolate both waveforms onto the common grid
    original_strain_interpolated = np.interp(t_common, target_strain[0], target_strain[1])
    solution_dummy_interpolated = np.interp(t_common, sol.t, dummy_h_strain)

    #finding and fixing the lag between the actual waveform and test waveforms
    # corr = np.correlate(original_strain_interpolated, solution_dummy_interpolated, mode='full')
    # lag = np.argmax(corr) - (len(solution_dummy_interpolated)-1)
    # dt = t_common[1] - t_common[0]
    # t_shifted = t_common + lag * dt

    # solution_aligned = np.interp(t_common, t_shifted, solution_dummy_interpolated, left=0.0, right=0.0)

    if plot:
        plt.plot(t_common, original_strain_interpolated, label='target')
        plt.plot(t_common, solution_dummy_interpolated, label='attempt')
        plt.legend()
        plt.show()

    target_norm = abs(original_strain_interpolated)

    difference_vector = abs(original_strain_interpolated) - abs(solution_dummy_interpolated)

    return np.linalg.norm(difference_vector) / np.linalg.norm(target_norm)

initial = [0.85, 0.2]

# M_values = np.arange(0.85, 1.2, 0.01)
# f_values = [f_min(M_value, 0.25, x0, experiment, False) for M_value in M_values]
# print(f_values)
# plt.plot(M_values, f_values)
# plt.yscale('log')
# plt.show()

res = minimize(f_min, initial, method='Nelder-Mead', args=(x0, experiment), options={'xatol': 1e-18, 'disp': True})
print(res.x, res.fun)
f_min(res.x, x0, experiment, True)

#Newton-CG
#Powell


