import numpy as np
import sympy as sp
import rpm_module as rpm

#### Build a phs structure ####
# Interconnexion matrix
S = np.array([[0, -1],
              [1, 0]])

# Symbols declaration
q, phi, C0, L0 = sp.symbols('q, phi, C_0, L_0', real=True)
# States variables
states = [q, phi]
# Additional parameters
parameters = [C0, L0]
# Hamiltonian expression


def hamiltonian(q, phi):
    return (q)**2 / (2*C0) + phi**2 / (2*L0)


H = hamiltonian(*states)

# Number of constraints
n_constraints = 0

# String with informations about the phs
about = "One capacitance and one inductance in a closed loop."

# Create dictionnary containing the structure
phs_struct = rpm.struct.build_struct_dict(S,
                                          H,
                                          states,
                                          parameters,
                                          n_constraints,
                                          about)


##### Save to a file #####
filename = 'models/pickles/linear_example_muller.pickle'
rpm.struct.store(phs_struct, filename)
