import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator

from solve_x_t import PNexapansion_x, E, F, ode, eval_function, pole_event, start, limit, t_span, step, x_vals, times, x0

dummy_E = PNexapansion_x()
dummy_F = PNexapansion_x()

def f_min(x0, M, nu, target_strain):
    #building the PNequations that will change with the minimization algorithm
    dummy_E.setM(M)
    dummy_E.setNu(nu)
    dummy_E.setPowers(E.p)
    dummy_E.setConstants(E.consts)
    dummy_dE_dx = dummy_E.differentiate()

    dummy_F.setM(M)
    dummy_F.setNu(nu)
    dummy_F.setPowers(F.p)
    dummy_F.setConstants(F.consts)

    dummy_F_x = F.get_Eq()

    #had to redefine the ODE equations in terms of these 'dummy' objects
    def ode(t, x):
        xx = float(x[0])
        y = -eval_function(dummy_F_x, xx)/eval_function(dummy_dE_dx, xx)

        return y

    sol = solve_ivp(ode, t_span, x0, events=pole_event, method='RK45', t_eval=np.linspace(start, limit, step))
    solution_dummy = sol.y[0] #just getting the x(t) values

    #interpolation so that each trial and the final output match

    # Find overlapping time interval
    t_min = max(times.min(), sol.t.min())
    t_max = min(times.max(), sol.t.max())

    # Choose a common grid
    N = max(len(times), len(sol.t))
    t_common = np.linspace(t_min, t_max, N)

    # Interpolate both waveforms onto the common grid
    orinal_x_interpolated = np.interp(t_common, times, x_vals)
    solution_dummy_interpolated = np.interp(t_common, sol.t, solution_dummy)

    return (orinal_x_interpolated - solution_dummy_interpolated)

print(f_min(x0, 1, 0.25, x_vals))
#print(type(E.p))
