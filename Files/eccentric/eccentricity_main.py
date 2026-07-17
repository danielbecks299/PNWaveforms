import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import newton
import time

start_time = time.time()

G, c, m1, m2 = 1, 1, 0.5, 0.5
M = m1 + m2
nu = (m1*m2)/M**2 

def E_xi(x, i):
    alpha = (-0.5 * (c**2) * M * nu * x)
    E0 = 1
    E1 = (5/4) - (2/i) - (nu/12)

    E = alpha * (E0 + E1*x)
    return E

def F_xi(x, i):
    alpha = (32 * ((c*x)**5) * nu**2)/(5*G * i**(3/2))
    F0 = (37/96) + (425/(96 * i**2)) - (61/(16*i))
    F1 = (139/112) + ((-5297/336) - (2725/384)*nu)/i + ((259*nu)/1152) + ((-289/3) + ((3605*nu)/384))/i**3 + ((1865/24) + ((3775/384)*nu))/i**2

    F = alpha * (F0 + F1*x)
    return F

def J_xi(x, i):
    alpha = ((G * M**2 * nu)/(c * jnp.sqrt(x)))
    J0 = jnp.sqrt(i)
    J1 = (((35/8) - (5*nu/4))/jnp.sqrt(i)) + (jnp.sqrt(i) * ((nu/4) - (5/8)))

    J = alpha * (J0 + J1*x)
    return J

def dJ_dt_xi(x, i):
    alpha = (32 * (c*nu)**2 * M * x**(7/2))/(5*i)
    dJ0 = -(7/8) + (15/(8*i))
    dJ1 = -(1597/2688) + ((-3125/128) - (275*nu/96))/(i**2) - (31*nu/32) + ((535/64) + (61*nu/8))/i

    dJ_dt = alpha * (dJ0 + dJ1*x)
    return dJ_dt

def a_t(x, i):
    alpha = 1/x
    a_t0 = 1
    a_t1 = (2/i) + (nu/3) - 3

    a_txi = alpha * (a_t0 + a_t1*x)
    return a_txi

def e_t(x, i):
    e_t0 = 1 - i 
    e_t1 = (-35/8) + (9*nu)/2 + i*((17/4) - ((13*nu)/6))

    e_txi = e_t0 + (e_t1*x)
    return e_txi

def e_r(x, i):
    e_r0 = 1 - i
    e_r1 = (3*nu)/2 - (3/4) + i*((5*nu)/6 - (15/4))

    e_rxi = e_r0 + (e_r1*x)
    return e_rxi

def e_phi_22(x, i):
    e_phi0 = 1 - i
    e_phi1 = (-3/4) + (5*nu)/2 + i*((-nu/6) - (15/8))

    e_phi_22xi = e_phi0 + (e_phi1*x)
    return e_phi_22xi

def H22(r, phi, x, i, t):
    alpha = (4*G*M*nu / c**4) * np.sqrt(np.pi/5) * np.exp(-1j * 2 * phi)
    dr = np.gradient(r, t)
    dphi = np.gradient(phi, t)

    H22_0 = (G*M)/r + (r**2 * dphi**2) + (2j * r * dr * dphi) - dr**2
    H22_1 = (1 / c**2) * (
        ((G*M/r)**2 * (nu/2 - 5)) 
        + (G*M*dr**2/r) * (-15/14 - 16*nu/7) 
        + (dr**4 * (27*nu/14 - 9/14)) 
        + (r * dr**3 * dphi * ((9j - 27j * nu)/7)) 
        + (G*M*r*dphi**2 * (11/42 + 26*nu/7)) 
        + ((dphi*r)**4 * (9/14 - 27*nu/14)) 
        + (dr * (G*M*dphi * (25j/21 + 45j*nu/7) + (r * dphi)**3 * (9j/7 - 27j*nu/7)))
    )

    return alpha * (H22_0 + H22_1*x)
            
def ode_xi(t, y):
    x, i = y

    dEdx = dE_dx(x, i)
    dEdi = dE_di(x, i)

    dJdx = dJ_dx(x, i)
    dJdi = dJ_di(x, i)

    F = F_xi(x, i)
    dJdt = dJ_dt_xi(x, i)

    denom = dEdx*dJdi - dJdx*dEdi

    #only flipped the sign on F
    dxdt = (-F*dJdi - dEdi*dJdt) / denom
    didt = (dEdx*dJdt + F*dJdx) / denom

    return [float(dxdt), float(didt)]

def invert_kepler(l, e_txi):
    f = lambda u: u - e_txi*np.sin(u) - l
    fp = lambda u: 1 - e_txi*np.cos(u)

    u = newton(f, x0=l, fprime=fp)

    return u

def find_omega(x):
    omega = (c**3 * x**(3/2))/(G * M)
    return omega

def i_event(t, y):
    x, i = y
    return i-1

i_event.terminal = True      # stop integration
i_event.direction = 1         # only trigger when i increases through 1

def denom_event(t, y):
    x, i = y
    D = (
        dE_dx(x, i)*dJ_di(x, i)
        - dJ_dx(x, i)*dE_di(x, i)
    )
    return D

denom_event.terminal = True
denom_event.direction = 0

#partial differentiaition with jax library
dE_dx = jit(grad(E_xi, argnums=0))
dE_di = jit(grad(E_xi, argnums=1))

dJ_dx = jit(grad(J_xi, argnums=0))
dJ_di = jit(grad(J_xi, argnums=1))

start = 0
limit = 500_000
step = 10_000_000
t_span = (start, limit)  

x0, i0 = 0.018, 0.7 #0.036, 0.7
y0 = [x0, i0]

start_time = time.time()

#solve ODE
sol_xi = solve_ivp(ode_xi, t_span, y0, method='RK45', events=[denom_event, i_event], rtol=1e-8, atol=1e-10, t_eval=np.linspace(start, limit, step))
t = sol_xi.t
x = sol_xi.y[0]
i = sol_xi.y[1]

#find l(t), mean anomaly
omega = find_omega(x)
dl_dt = (omega*i)/((3*x + i))
l = cumulative_trapezoid(dl_dt, t, initial=0.0)

#use this to find the eccentricity
e_txi = np.sqrt(e_t(x, i))
u = invert_kepler(l, e_txi)

#find r(t)
e_rxi = np.sqrt(e_r(x, i))
r = a_t(x,i) * (1 - (e_rxi * np.cos(u)))

#find the eccentric phi and regular phi
ephi = np.sqrt(e_phi_22(x, i))
raw_v = np.arctan2(np.sqrt(1 + ephi) * np.sin(u/2), np.sqrt(1 - ephi) * np.cos(u/2))
v_xi = 2 * np.unwrap(raw_v)
phi_xi = (1 + (3*x)/i) * v_xi

#plotting the 2,2 mode
wave = H22(r, phi_xi, x, i, t)

#cartesian coords
r1 = r * (m2 / M)
x_coords = r1 * np.cos(phi_xi)
y_coords = r1 * np.sin(phi_xi)

r2 = r * (m1 / M)
x_coords2 = -r2 * np.cos(phi_xi)
y_coords2 = -r2 * np.sin(phi_xi)

plt.plot(t, i, label='iota')
plt.plot(t, x, label='x')
plt.xlabel(r"$Time, t$")
plt.legend(title=f"$m_1 ={m1}, m_2 = {m2}, x_0 = {x[0]}, i_0 = {i[0]}$, 1st Order")
plt.show()

plt.plot(t, l)
plt.xlabel(r"$Time, t$")
plt.ylabel(r"$Mean Anomaly, \ell$")
plt.legend(title=f"$m_1 ={m1}, m_2 = {m2}, x_0 = {x[0]}, i_0 = {i[0]}$, 1st Order")
plt.show()

plt.plot(t, r, label='r(t)')
plt.xlabel(r"$Time, t$")
plt.ylabel(r"$r$")
plt.legend(title=f"$m_1 ={m1}, m_2 = {m2}, x_0 = {x[0]}, i_0 = {i[0]}$, 1st Order")
plt.show()

plt.plot(t, np.real(wave))
plt.xlabel(r"$Time, t$")
plt.ylabel(r"$h$")
plt.legend(title=f"$m_1 ={m1}, m_2 = {m2}, x_0 = {x[0]}, i_0 = {i[0]}$, 1st Order")
plt.show()

plt.plot(x_coords, y_coords, label='Mass 1')
plt.plot(x_coords2, y_coords2, label="Mass 2")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend(title=f"$m_1 ={m1}, m_2 = {m2}, x_0 = {x[0]}, i_0 = {i[0]}$, 1st Order")
plt.show()

print(sol_xi.nfev)
print(sol_xi.njev)
print(sol_xi.status)
print(sol_xi.message)

print(time.time() - start_time)


