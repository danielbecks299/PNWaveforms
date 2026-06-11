#solve for x(t) from E(x) & F(x) -> only first 2 terms

#General notes: powers must be set before constants

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#goal here is to build each equation E(x) and F(x) as indivudual arrays with coefficients and powers
class PNexapansion_x:
    def __init__(self, m1, m2, r, c=1, G=1): #when object is created speed of light (c), grav. const(G) are initialized, nu as well
        self.m1, self.m2, self.r, self.c, self.G = m1, m2, r, c, G
        self.nu = (m1*m2)/(m1+m2)**2            

    def setPowers(self, p):     #p is a a real number that gets interated into powers of x
        if isinstance(p, tuple):
            self.p = np.array(p)
        else:
            self.p = np.zeros(p+1)  #array of powers of x
            for P in range(p+1):    #arrays start at zero, we want x^0+...+x^n
                self.p[P] = P
        
        return self.p

    def setConstants(self, consts, alpha=1):     #takes array of constants
        self.consts = np.asarray(consts, dtype=np.longdouble)
        self.consts *= alpha
        const_len = len(self.consts)
        powers_len = len(self.p)

        #ensures we have equal powers and constants, though we must manually enter 1 as the first constant
        msg = f"Error: {const_len} constants and {powers_len} powers"

        if (len(self.consts) != len(self.p)):
            raise ValueError(msg)

        return self.consts
    
    def differentiate(self): #this is the differentiation  
        self.new_p = self.p[1:] - 1 #discard the first element, so i always need to input a zeroth power, even if the coefficient is 0
        self.new_consts = self.consts[1:] * self.p[1:]

        return np.array([self.new_p, self.new_consts], dtype=np.longdouble)
    
    def get_Eq(self):       #combines arrays and reshapes equation into n x 2 matrix
        y = np.array([self.p, self.consts], dtype=np.longdouble)

        return y
    
def eval_function(powers_const, x):
    y = 0
    y_big = np.longdouble(y)

    for j in range(len(powers_const[0])):
        y_big += powers_const[1,j]*(x**powers_const[0,j])

    return y_big

#from down on its setting parameters which gets super SUPER messy

m1 = 1
m2 = 1
r = 1

E = PNexapansion_x(m1, m2, r)    #create energy equation object
F = PNexapansion_x(m1, m2, r) 

#initialize constants from PNpedia
E_0 = 1
E_1 = (-3/4)-(E.nu/12)
E_2 = (-27/8)+(19*E.nu/8)-((E.nu**2)/24)

#build energy equation
E.setPowers(3) #this however is only second order due to the common factor of x
E.setConstants((0, E_0, E_1, E_2), alpha = -(0.5)*(E.c)**2*(E.m1+E.m2)*E.nu)

powers_F = (5, 6, 6.5, 7)
F_0 = 1
F_1 = -(1247/336)-((35*F.nu)/12)
F_2 = 4*np.pi
F_3 = -(44711/9072)+(9271*F.nu/504)+(65*(F.nu**2)/18)
F_4 = -(8191*np.pi/672)-(583*np.pi*F.nu/24)

F.setPowers(powers_F)
F.setConstants((F_0, F_1, F_2, F_3), alpha = 32*(F.c**5)*(F.nu**2)/(5*F.G))

#4x2 matrices of constants and powers
E_x = E.get_Eq()
dE_dx = E.differentiate()
F_x = F.get_Eq()

#solve ODE
def ode(t, x):
    xx = float(x[0])

    y = -eval_function(F_x, xx)/eval_function(dE_dx, xx)

    return y

#to deal with the root where dE/dx crosses 0
def pole_event(t, x):
    return eval_function(dE_dx, x[0])

pole_event.terminal = True
pole_event.direction = 0

x0 = [0.1]
limit = 2000
t_span = (0, limit)  
solution = solve_ivp(ode, t_span, x0, events=pole_event, method='RK45', t_eval=np.linspace(0, limit, 10000))

times = solution.t
values = solution.y[0]
x = np.linspace(0, 0.3, 15)

fig, ax = plt.subplots()
ax.set_xlabel(r"$Time, t$")
ax.set_ylabel(r"$x=(M\Omega)^{2/3}$")
ax.legend(title=r"$m_1 = m_2 = 1, c = 1, G = 1, x_0 = 0.1$, 2nd Order")

plt.plot(times, values, label='x(t)')
plt.show()