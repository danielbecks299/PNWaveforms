#solve for x(t) from E(x) & F(x) -> only first 2 terms

#General notes: powers must be set before constants

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#goal here is to build each equation E(x) and F(x) as indivudual arrays with coefficients and powers
class PNexapansion_x:
    def __init__(self, M, r, c=1, G=1): #when object is created speed of light (c), grav. const(G) are initialized, nu as well
        self.M, self.r, self.c, self.G = M, r, c, G
        self.nu = (2*M)/M**2            #assume equal masses for now

    def setPowers(self, p):     #p is a a real number that gets interated into powers of x
        if isinstance(p, np.ndarray):
            self.p = np.array(p)
        else:
            self.p = np.zeros(p+1)  #array of powers of x
            for P in range(p+1):    #arrays start at zero, we want x^0+...+x^n
                self.p[P] = P
        
        return self.p

    def setConstants(self, consts, alpha=1):     #takes array of constants
        self.consts = np.asarray(consts)
        self.consts *= alpha
        const_len = len(self.consts)
        powers_len = len(self.p)

        #ensures we have equal powers and constants, though we must manually enter 1 as the first constant
        msg = f"Error: {const_len} constants and {powers_len} powers"

        if (len(self.consts) != len(self.p)):
            raise ValueError(msg)

        return self.consts
    
    def differentiate(self): #this is the differentiation
        if len(self.p) <= 1:  
            return np.array([0]), np.array([0])
        
        self.new_p = self.p[1:] - 1
        self.new_consts = self.consts[1:] * self.p[1:]

        return self.new_p, self.new_consts
    
    def get_Eq(self):       #combines arrays and reshapes equation into n x 2 matrix
        y = self.p, self.consts
        return np.array(y)
    
def eval_function(powers_const, x):
    y = 0

    for j in range(len(powers_const[0])):
        y += powers_const[1,j]*(x**powers_const[0,j])

    return y

#from down on its setting parameters which gets super SUPER messy

E = PNexapansion_x(M=1, r=1)    #create energy equation object
F = PNexapansion_x(M=1, r=1)

#initialize constants from PNpedia
E_1 = 1
E_2 = -(3/4)-(E.nu/12)
E_3 = -(27/8)+(19*E.nu/8)-(E.nu**2/24)

#build energy equation
E.setPowers(int(3))
E.setConstants((0,E_1, E_2, E_3), alpha = (-0.5)*E.M*E.nu*(E.c**2))


powers_F = np.array((5,6,5.5,7,7.5))
F_1 = 1
F_2 = -(1247/336)-((35*F.nu)/12)
F_3 = 4*np.pi
F_4 = -(44711/9072)+(9271*F.nu/504)-(65*(F.nu**2)/18)
F_5 = -(8191*np.pi/672)-(583*np.pi*F.nu/24)

F.setPowers(powers_F)
F.setConstants((F_1,F_2,F_3,F_4,F_5), alpha = (32*F.c**5*F.nu**2)/5*F.G)


#4x2 matrices of constants and powers
E_x = E.get_Eq()
dE_dx = np.array(E.differentiate())
F_x = F.get_Eq()

#solve ODE
def ode(t, u):
    x = u[0]
    y = eval_function(F_x, x)/eval_function(dE_dx, x)
    return y

x0 = [0.1]
t_span = (0, 50)  
solution = solve_ivp(ode, t_span, x0, method='RK45', t_eval=np.linspace(0, 50, 100))

times = solution.t
values = solution.y[0]

plt.plot(times, values, label='x(t)')
plt.legend()
plt.show()