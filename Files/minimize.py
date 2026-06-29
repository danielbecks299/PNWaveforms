import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize

from solve_x_t import PNexapansion_x, E, F, ode, eval_function, pole_event, limit, t_span, step, E_0, E_1, E_2, F_0, F_1, F_2, F_3
from find_h_t import H_2_2_X, PSI_X, imganinary_exponent, experiment, h_strain_22

def f_min(x0, M, nu, target_strain=experiment):
    #building the PNequations that will change with the minimization algorithm
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
    t_common = np.linspace(t_min, t_max, N)

    # Interpolate both waveforms onto the common grid
    original_x_interpolated = np.interp(t_common, target_strain[0], target_strain[1])
    solution_dummy_interpolated = np.interp(t_common, sol.t, dummy_h_strain)

    corr = np.correlate(original_x_interpolated, solution_dummy_interpolated, mode='full')
    lag = np.argmax(corr) - (len(solution_dummy_interpolated)-1)
    solution_aligned = np.roll(solution_dummy_interpolated, lag)

    difference_vector = original_x_interpolated - solution_aligned

    return np.linalg.norm(difference_vector)

x0 = [0.05]
res = minimize(f_min, x0, method='Nelder-Mead', args=(1, 0.25, experiment), options={'xatol': 1e-8, 'disp': True})
print(res.x)