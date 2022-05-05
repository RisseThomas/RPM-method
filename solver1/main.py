import numpy as np
import rpm_module as rpm
import sympy as sp

# Problem data
# Load a phs stucture
filename = "models/pickles/linear_example_muller.pickle"
phs_struct = rpm.struct.load(filename)

# Replace parameters of the hamiltonian with their desired values
parameters = phs_struct["Parameters"]
print(parameters)
C0 = parameters[0]
# C2 = parameters[1]
L0 = parameters[1] 
phs_struct["H"] = phs_struct["H"].subs([(C0, 1), (L0, 1)])
phs_struct["H"]


# Solver parameters
p_order = 1
k_order = 2
sr = 10
step_size = 1/sr
quad_order = 50
epsilon = 10**(-10)
maxIter = 10

solver = rpm.RPMSolverPHS(phs_struct, p_order, k_order, step_size,
                          quad_order, epsilon, maxIter)

# Initialization
init = np.ones(len(phs_struct["S"]), dtype=np.float64)

duration = 1
t = np.linspace(0, duration, int(duration/step_size))
x, dx_proj, l_mults, dx_regul = solver.simulate(init, duration)
