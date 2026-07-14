import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import sympy as sp
import time

start_time = time.time()

G, c, m1, m2 = 1, 1, 0.5, 0.5
M = m1 + m2
nu = (m1*m2)/(m1+m2)**2 

def E_xi(x, i):
    alpha = (-0.5 * (c**2) * M * nu * x)
    E0 = 1
    E1 = (5/4) - (2/i) - (nu/12)
    E2 = (5/8) + (5/i**2) + ((5-(2*nu))/jnp.sqrt(i)) + ((-5+(nu/3))/i) - ((5*nu)/8) - (nu**2/24)

    E = alpha * (E0 + E1*x)
    return E

def F_xi(x, i):
    alpha = (32 * ((c*x)**5) * nu**2)/(5*G * i**(3/2))
    F0 = (37/96) + (425/(96 * i**2)) - (61/(16*i))
    F1 = (139/112) + ((-5297/336) - (2725/384)*nu)/i + ((259*nu)/1152) + ((-289/3) + ((3605*nu)/384))/i**3 + ((1865/24) + ((3775/384)*nu))/i**2
    F1_5 = 0
    F2 = 0

    F = alpha * (F0 + F1*x)
    return F

def J_xi(x, i):
    alpha = ((G * M**2 * nu)/(c * jnp.sqrt(x)))
    J0 = jnp.sqrt(i)
    J1 = (((38/5) - (5*nu/4))/jnp.sqrt(i)) + (jnp.sqrt(i) * ((nu/4) - (5/8)))
    J2 = 0 #fill this in later

    J = alpha * (J0 + J1*x)
    return J

def dJ_dt_xi(x, i):
    alpha = (32 * (c*nu)**2 * M * x**(7/2))/(5*i)
    dJ0 = -(7/8) + (15/(8*i))
    dJ1 = -(1597/2688) + ((-3125/128) - (275*nu/96))/(i**2) - (31*nu/32) + ((535/64) + (61*nu/8))/i
    dJ1_5 = 0
    dJ2 = 0

    dJ_dt = -alpha * (dJ0 + dJ1*x)
    return dJ_dt

def ode(t, y):
    x, i = y

    dEdx = dE_dx(x, i)
    dEdi = dE_di(x, i)

    dJdx = dJ_dx(x, i)
    dJdi = dJ_di(x, i)

    F = F_xi(x, i)
    dJdt = dJ_dt_xi(x, i)

    denom = dEdx*dJdi - dJdx*dEdi

    dxdt = (F*dJdi - dEdi*dJdt) / denom
    didt = (dEdx*dJdt - F*dJdx) / denom

    return [float(dxdt), float(didt)]

def i_event(t, y):
    x, i = y
    return i - 1

i_event.terminal = True      # stop integration
i_event.direction = 1         # only trigger when i increases through 1

#partial differentiaition with jax library
dE_dx = jit(grad(E_xi, argnums=0))
dE_di = jit(grad(E_xi, argnums=1))

dJ_dx = jit(grad(J_xi, argnums=0))
dJ_di = jit(grad(J_xi, argnums=1))

start = 0
limit = 500_00
step = 10_000_000
t_span = (start, limit)  

x0, i0 = 0.0417079949301, 0.13
y0 = [x0, i0]

start_time = time.time()

sol = solve_ivp(ode, t_span, y0, method='RK45', events=i_event, t_eval=np.linspace(start, limit, step))
t = sol.t
x = sol.y[0]
i = sol.y[1]

plt.plot(t, i, label='iota')
plt.plot(t, x, label='x')
#plt.plot(t_plot, y_plot[1], label="iota")
plt.legend()
plt.show()

print(sol.nfev)
print(sol.njev)
print(sol.status)
print(sol.message)

print(time.time() - start_time)


