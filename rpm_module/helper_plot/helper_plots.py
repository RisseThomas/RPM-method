import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rpm_module as rpm


def plot_basis(solver):
    """Plots both basis of the solver in two different subplots.

    Args:
        solver (object): instance of the solver class

    Returns:
        object: matplotlib figure
    """
    evaluate_proj = solver.basis.evaluate_proj
    evaluate_regul = solver.basis.evaluate_regul
    x = np.linspace(0, 1, 500)
    y_proj = evaluate_proj(x)

    fig = plt.figure(figsize=(12, 6))
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
    return fig


def plot_basis_P(solver):
    """Plots the basis of the projector P.

    Args:
        solver (object): instance of the solver class

    Returns:
        object: matplotlib figure
    """
    # Evaluating basis function values
    evaluate_proj = solver.basis.evaluate_proj
    x = np.linspace(0, 1, 500)
    y_proj = evaluate_proj(x)

    # Creating figure and colormap
    fig = plt.figure(figsize=(12, 6))
    plt.title("Basis functions of the projector P")
    plt.xlim((-1, 2))
    miny = np.min(y_proj)*1.2
    maxy = np.max(y_proj)*1.2
    plt.ylim(miny, maxy)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(y_proj))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.autumn)

    # Add background colors
    plt.axvspan(-1, 0, facecolor='pink', alpha=0.1)
    plt.axvspan(1, 2, facecolor='pink', alpha=0.1)

    for n in range(len(y_proj)):
        i = len(y_proj)-1-n
        alpha = 1 - i/(3*len(y_proj))
        color = cmap.to_rgba(i)
        if i == 0:
            lw = 3
        else:
            lw = 2
        plt.plot(x, y_proj[i], label=f'Basis function {i+1}',
                 c=color, alpha=alpha, linewidth=lw)
        plt.plot(x-1, y_proj[i], c=color, alpha=alpha-0.2, linewidth=lw)
        plt.plot(x+1, y_proj[i], c=color, alpha=alpha-0.2, linewidth=lw)
    
    # Add vertical lines to separate time frames
    plt.vlines([0, 1], [miny, miny], [maxy, maxy], colors='black',
               linestyles='dashed')

    # Label and legends
    plt.xlabel('$ tau $')
    plt.legend()

    return fig


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


def plot_phase_diagram_2D(solver, x, dx_proj, dx_regul, N_points=10):
    # First we evaluate integration of the basis functions at intermediate
    # points.
    taui = np.linspace(0, 1, N_points, endpoint=False)
    integrals = np.zeros((N_points, solver.k_order + solver.p_order))
    for i, tau in enumerate(taui):
        integrals[i] = solver.quad.integrate(solver.basis.evaluate_all, 0, tau)
    integrals_P = integrals[:, :solver.p_order]

    # Concatenation of flows
    dx = np.concatenate((dx_proj, dx_regul), axis=2)

    steps = len(x)
    int_states_regul = np.zeros((len(x) * N_points, solver.n_state))
    int_states_non_regul = np.zeros((len(x) * N_points, solver.n_state))
    for step in range(steps):
        first_ind = step*N_points
        last_ind = first_ind+N_points
        x0 = x[step]
        for state in range(solver.n_state):
            int_states_regul[first_ind:last_ind, state] = \
                x0[state] + solver.time_step * dx[step, state]@integrals.T

            int_states_non_regul[first_ind:last_ind, state] = \
                x0[state] +\
                solver.time_step * dx_proj[step, state]@integrals_P.T

    fig = plt.figure(figsize=(12, 6))
    plt.suptitle("Phase diagram")

    # Without regularization
    plt.subplot(1, 2, 1)

    plt.plot(int_states_non_regul[:, 0], int_states_non_regul[:, 2])
    x0_points = np.arange(0, len(x)*N_points, N_points)
    plt.scatter(int_states_non_regul[x0_points, 0],
                int_states_non_regul[x0_points, 2], c='r')

    plt.xlabel(solver.states[0])
    plt.ylabel(solver.states[1])
    plt.title("Without regularization")

    # With regularization
    plt.subplot(1, 2, 2)

    plt.plot(int_states_regul[:, 0], int_states_regul[:, 2])
    x0_points = np.arange(0, len(x)*N_points, N_points)
    plt.scatter(int_states_regul[x0_points, 0],
                int_states_regul[x0_points, 2], c='r')

    plt.xlabel(solver.states[0])
    plt.ylabel(solver.states[1])
    plt.title("With regularization")

    plt.tight_layout()

    return fig


def plot_gradients_phase(solver, x, dx_proj, dx_regul, N_points=10):
    # First we evaluate integration of the basis functions at intermediate
    # points.
    taui = np.linspace(0, 1, N_points, endpoint=False)
    integrals = np.zeros((N_points, solver.k_order + solver.p_order))
    for i, tau in enumerate(taui):
        integrals[i] = solver.quad.integrate(solver.basis.evaluate_all, 0, tau)
    integrals_P = integrals[:, :solver.p_order]

    # Concatenation of flows
    dx = np.concatenate((dx_proj, dx_regul), axis=2)

    steps = len(x)
    int_states_regul = np.zeros((len(x) * N_points, solver.n_state))
    int_states_non_regul = np.zeros((len(x) * N_points, solver.n_state))
    for step in range(steps):
        first_ind = step*N_points
        last_ind = first_ind+N_points
        x0 = x[step]
        for state in range(solver.n_state):
            int_states_regul[first_ind:last_ind, state] = \
                x0[state] + solver.time_step * dx[step, state]@integrals.T

            int_states_non_regul[first_ind:last_ind, state] = \
                x0[state] +\
                solver.time_step * dx_proj[step, state]@integrals_P.T

    # Computation of gradients from the state
    gradients_non_regul = solver.gradients(int_states_non_regul)
    gradients_regul = solver.gradients(int_states_regul)

    fig = plt.figure(figsize=(12, 6))
    plt.suptitle("Gradient phase diagram")

    # Without regularization
    plt.subplot(1, 2, 1)

    plt.plot(gradients_non_regul[:, 0], gradients_non_regul[:, 1])
    x0_points = np.arange(0, len(x)*N_points, N_points)
    plt.scatter(gradients_non_regul[x0_points, 0],
                gradients_non_regul[x0_points, 1], c='r')

    plt.xlabel(f'GradH / {solver.states[0]}')
    plt.ylabel(f'GradH / {solver.states[1]}')
    plt.title("Without regularization")

    # With regularization
    plt.subplot(1, 2, 2)

    plt.plot(gradients_regul[:, 0], gradients_regul[:, 1])
    x0_points = np.arange(0, len(x)*N_points, N_points)
    plt.scatter(gradients_regul[x0_points, 0],
                gradients_regul[x0_points, 1], c='r')

    plt.xlabel(f'GradH / {solver.states[0]}')
    plt.ylabel(f'GradH / {solver.states[1]}')
    plt.title("With regularization")

    plt.tight_layout()

    return fig
