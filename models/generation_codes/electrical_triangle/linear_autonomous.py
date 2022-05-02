import numpy as np
import sympy as sp
import rpm_module as rpm

#### Build a phs structure ####
# Interconnexion matrix
S = np.array([[0, 0, -1, -1],
              [0, 0, 0, -1],
              [1, 0, 0, 0],
              [1, 1, 0, 0]])

# Symbols declaration
q1, q2, phi, C1, C2, L0 = sp.symbols('q1, q2, phi, C_1, C_2, L_0', real=True)
# States variables
states = [q1, q2, phi]
# Additional parameters
parameters = [C1, C2, L0]
# Hamiltonian expression


def hamiltonian(q1, q2, phi):
    return (q1)**2 / (2*C1) + (q2)**2 / (2*C2) + phi**2 / (2*L0)


H = hamiltonian(*states)

# Number of constraints
n_constraints = 1

# String with informations about the phs
about = "This simple system is composed of two inductances and one\
capacitance.\n They are all linked to mass and to each other at te other end."

# Create dictionnary containing the structure
phs_struct = rpm.struct.build_struct_dict(S,
                                          H,
                                          states,
                                          parameters,
                                          n_constraints,
                                          about)


##### Save to a file #####
filename = 'models/pickles/linear_autonomous_triangle.pickle'
rpm.struct.store(phs_struct, filename)
