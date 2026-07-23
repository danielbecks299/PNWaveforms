from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import differential_evolution
from scipy.signal import correlate
import numpy as np
import time

from eccentricity_main import trial, start, limit, step, t_span
from eccentricity_main import e_t, e_r, e_phi_22, a_t, H22
from eccentricity_main import ode_xi, find_omega, invert_kepler, denom_event, i_event

start_time = time.time()

def f_min(parameters, target_strain=trial, plot=False, return_data=False):
    G, c= 1, 1
    M = parameters[0]
    nu = parameters[1]
    y0 = (parameters[2], parameters[3])

    sol_xi = solve_ivp(ode_xi, t_span, y0, method='BDF', events=[denom_event, i_event], rtol=1e-8, atol=1e-10, t_eval=np.linspace(0, 30_000, 300_000))
    t = sol_xi.t
    x = sol_xi.y[0]
    i = sol_xi.y[1]

    if not sol_xi.success or len(sol_xi.t) < 2:
        return np.inf

    omega = find_omega(x, M)
    dl_dt = (omega*i)/((3*x + i))
    l = cumulative_trapezoid(dl_dt, t, initial=0.0)

    et = e_t(x, i, M, nu)
    er = e_r(x, i, M, nu)
    ep = e_phi_22(x, i, M, nu)

    if np.any(et < 0) or np.any(er < 0) or np.any(ep < 0):
        return 1e20

    e_txi = np.sqrt(et)
    e_rxi = np.sqrt(er)
    ephi = np.sqrt(ep)

    #use this to find the eccentricity
    u = invert_kepler(l, e_txi)

    #find r(t)
    r = a_t(x,i, M, nu) * (1 - (e_rxi * np.cos(u)))

    K = 1.0 + (3.0 * x / i)
    phi_dot = (K * dl_dt * np.sqrt(1.0 - ephi**2) / ((1.0 - e_txi * np.cos(u)) * (1.0 - ephi * np.cos(u))))
    phi_xi = cumulative_trapezoid(phi_dot, t, initial=0.0)

    #plotting the 2,2 mode
    wave_test = np.real(H22(r, phi_xi, t))

    #interpolation
    t_common = target_strain[0]
    original_strain_interpolated = np.interp(t_common, target_strain[0], target_strain[1])
    solution_dummy_interpolated = np.interp(t_common, t, wave_test)

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
    target_norm = np.abs(original_strain_interpolated)
    difference_vector = np.abs(original_strain_interpolated) - np.abs(solution_aligned)

    return np.linalg.norm(difference_vector) / np.linalg.norm(target_norm)

bounds = [
    (0.8, 1.2),      # M
    (0.1, 0.25),     # nu
    (0.01, 0.014),  # x0
    (0.59, 0.9)      # i0
]

result = differential_evolution(f_min, bounds, maxiter=100, popsize=15, polish=False)
print(result.x, result.fun)

print("--- %s seconds ---" % (time.time() - start_time))