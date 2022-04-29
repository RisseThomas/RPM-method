import numpy as np
import rpm_module as rpm
import sympy as sp

# Problem data

# Interconnexion matrix
S = np.array([[0, -1],
             [1, 0]])

# Hamiltonian
q, phi, C0, phi0 = sp.symbols('q, phi, C_0, phi_0', real=True)
states = [q, phi]


def hamiltonian(q, phi):
    return q**2 / (2*C0) + phi**2 / (2*phi0)


H = hamiltonian(q, phi)
H = H.subs(C0, 1)
H = H.subs(phi0, 1)


# Solver parameters
p_order = 1
k_order = 2
sr = 10
step_size = 1/sr
quadOrder = 50
epsilon = 10**(-10)
maxIter = 10

solver = rpm.RPMSolverPHS(S, H, states, p_order, k_order, step_size,
                          quadOrder, epsilon, maxIter)

# Initialization
init = np.ones(len(S), dtype=np.float64)

duration = 1
t = np.linspace(0, duration, int(duration/step_size))
x, dx_proj, dx_regul = solver.simulate(init, duration)
