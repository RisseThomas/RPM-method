import numpy as np
import matplotlib.pyplot as plt
import rpm_module as rpm


def plot_basis(solver):
    evaluate_proj = solver.basis.evaluate_proj
    evaluate_regul = solver.basis.evaluate_regul
    x = np.linspace(0, 1, 500)
    y_proj = evaluate_proj(x)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Basis functions")

    plt.subplot(1, 2, 1)
    plt.title("Functions of the projection step")
    for n in range(len(y_proj)):
        plt.plot(x, y_proj[n], label=f'Basis function {n}')
        plt.plot(x-1, y_proj[n])
        plt.plot(x+1, y_proj[n])
    plt.xlabel('$ tau $')
    plt.legend()

    x2 = np.linspace(-1, 0, 500)
    y_regul = evaluate_regul(x)
    plt.subplot(1, 2, 2)
    plt.title("Basis functions of the regularization step")
    for n in range(int(len(y_regul)/2)):
        i_alpha_1 = n+int(len(y_regul)/2)
        plt.plot(x, y_regul[n], label=f'Basis function {n}')
        plt.plot(x, y_regul[i_alpha_1], label=f'Basis function {n}')
        plt.plot(x2, y_regul[n], label=f'Basis function {i_alpha_1}',
                 linestyle="--")
        plt.plot(x2, y_regul[i_alpha_1], label=f'Basis function {i_alpha_1}',
                 linestyle="--")
    plt.xlabel('$ tau $')
    plt.legend()

    plt.tight_layout()


def plot_gradient_projections(solver, x0, dx):
    tau = np.linspace(0, 1, 50)
    unproj_gradients, proj_gradient_coeffs, sum_proj_gradients = \
        rpm.proj.check_proj_gradient(solver, x0, dx, tau)
    n_state = solver.n_state
    n_basis = solver.p_order

    plt.figure(figsize=(12, 12))
    plt.suptitle("Gradients projetés et non projetés")

    for statei in range(n_state):
        plt.subplot(2, n_state, statei+1)
        plt.xlabel('Tau')
        plt.plot(tau, unproj_gradients[:, statei],
                 label='Unprojected gradient', linewidth=4, c='r')

        plt.plot(tau, sum_proj_gradients[:, statei],
                 label='Sum of projected gradients', linestyle='', marker='x',
                 markersize=15, c='g')
        plt.legend()
        plt.title(f"State {statei}")
        ax = plt.subplot(2, n_state, n_state+statei+1)
        plt.xlabel('Basis function')
        plt.ylabel('Projection coefficient')
        ax.set_yscale('log')
        plt.bar(np.arange(n_basis), np.abs(proj_gradient_coeffs[statei]))
    plt.tight_layout()


def plot_gradients(solver, x, labels):
    x = np.tile(x, (solver.n_state, 1)).T
    print(x.shape)
    gradients = np.zeros_like(x)
    for i, xi in enumerate(x):
        gradients[i] = solver.gradients(xi)
    plt.figure()
    plt.title("Flows as functions of states")
    for i, gradient in enumerate(gradients.T):
        plt.plot(x[:, 0], gradient, label=labels[i])
    plt.legend()


def plot_error_energy(solver, x, t):
    plt.figure()
    plt.title("Error on stored Energy")
    plt.plot(t[:-1], solver.H(x) - solver.H(x)[0])
    plt.xlabel("Time")
    plt.ylabel("Error on energy conservation")


def plot_flows_trajectories(solver, dx_proj, dx_regul, N_points=10):
    """Plot flows evolution using resynthesis operation
    to obtain flows values during frames from projection
    coefficients.

    Args:
        solver (object): instance of the RPM solver class
        dx_proj (array): array of projected flows coefficients of the
                        projection step.
                        Size : [n, solver.n_state, solver.p_order]
        dx_regul (array): array of projected flows coefficients of the
                        regularization step.
                        Size : [n, solver.n_state, solver.k_order]
        N_points (int): number of points per frame for the plot.
    """
    # Number of frames
    steps = len(dx_proj)
    # Intermediate points
    tau = np.linspace(0, 1, N_points)
    # Computation of basis function values at intermediate points
    proj_points = solver.basis.evaluate_proj(tau)
    regul_points = solver.basis.evaluate_regul(tau)
    # Synthesis
    synth_proj = np.zeros((steps, solver.n_state, N_points))
    synth_regul = np.zeros((steps, solver.n_state, N_points))
    for step in range(steps):
        synth_proj[step] = dx_proj[step] @ proj_points
        synth_regul[step] = dx_regul[step] @ regul_points

    # Full synthesis
    synth_full = synth_proj + synth_regul

    # We want to plot trajectories using recontruction
    # from the projection coefficients
    plt.figure(figsize=(12, 6))
    for state in range(solver.n_state):
        plt.subplot(1, solver.n_state, state+1)
        for step in range(steps):
            if step == 0:
                plt.plot(step+tau, synth_proj[step, state], color='r',
                         label='Projection step')
                plt.plot(step+tau, synth_full[step, state], color='b',
                         label='Both steps')
            else:
                plt.plot(step+tau, synth_proj[step, state], color='r')
                plt.plot(step+tau, synth_full[step, state], color='b')
        plt.legend()
