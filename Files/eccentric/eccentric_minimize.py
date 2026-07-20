from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import differential_evolution
import numpy as np

from eccentricity_main import trial, start, limit, step, t_span
from eccentricity_main import E_xi, F_xi, J_xi, dJ_dt_xi, dE_di_func, dE_dx_func, dJ_di_func, dJ_dx_func
from eccentricity_main import e_t, e_r, e_phi_22, a_t, H22
from eccentricity_main import ode_xi, find_omega, invert_kepler, denom_event, i_event

def f_min(parameters, target_strain=trial, plot=False, return_data=False):
    G, c, m1, m2 = 1, 1, 0.5, 0.5
    M = m1 + m2
    nu = (m1*m2)/M**2 
    y0 = parameters

    sol_xi = solve_ivp(ode_xi, t_span, y0, method='BDF', events=[denom_event, i_event], rtol=1e-8, atol=1e-10, t_eval=np.linspace(start, limit, step))
    t = sol_xi.t
    x = sol_xi.y[0]
    i = sol_xi.y[1]

    if not sol_xi.success or len(sol_xi.t) < 2:
        return 1e20

    omega = find_omega(x)
    dl_dt = (omega*i)/((3*x + i))
    l = cumulative_trapezoid(dl_dt, t, initial=0.0)

    et = e_t(x, i)
    er = e_r(x, i)
    ep = e_phi_22(x, i)

    if np.any(et < 0) or np.any(er < 0) or np.any(ep < 0):
        return 1e20

    e_txi = np.sqrt(et)
    e_rxi = np.sqrt(er)
    ephi = np.sqrt(ep)

    #use this to find the eccentricity
    u = invert_kepler(l, e_txi)

    #find r(t)
    r = a_t(x,i) * (1 - (e_rxi * np.cos(u)))

    K = 1.0 + (3.0 * x / i)
    phi_dot = (K * dl_dt * np.sqrt(1.0 - ephi**2) / ((1.0 - e_txi * np.cos(u)) * (1.0 - ephi * np.cos(u))))
    phi_xi = cumulative_trapezoid(phi_dot, t, initial=0.0)

    #plotting the 2,2 mode
    wave_test = np.real(H22(r, phi_xi, t))

    #interpolation
    t_common = trial[0]
    original_strain_interpolated = np.interp(t_common, target_strain[0], target_strain[1])
    solution_dummy_interpolated = np.interp(t_common, t, wave_test)
    
    difference_vector = np.abs(original_strain_interpolated) - np.abs(solution_dummy_interpolated)
    return np.linalg.norm(difference_vector)

bounds = [(0.01, 0.019), (0.5, 0.9)]

result = differential_evolution(f_min, bounds, maxiter=100, popsize=15, polish=False)
print(result.x, result.fun)