from path import base_path
import sys
sys.path.append(base_path)
from path import qsl_path, dynamics_path
from cost_function import get_ARP_hamiltonian
import time_evolving_mpo as tempo
from generate_adjoint_pt import generate_transmon_PT
from cost_function import qubit_reset_cost_function, qubit_reset_gradient
from dimitrios_methods import load_process_tensor
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

import numpy as np

optimisation_name = 'transmon_reset'

generate_pt = False
do_optimisation = True
generate_dynamics = True
plot_results = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ parameters ~~~~~~~~
dt = 0.01  #10ps
dkmax = 80
esprel = 10**(-6)
temp = 0.01 # Kelvin
max_time = 5 #ns

initial_state = tempo.operators.spin_dm('mixed')

if generate_pt:
    generate_transmon_PT(
        dt=dt,
        dkmax=dkmax,
        esprel=esprel,
        _temperature=temp,
        max_time=max_time
    )

if do_optimisation:
    pt = load_process_tensor(
                    max_time,
                    dt,
                    dkmax,
                    esprel,
                    temp)

    times = pt.get_expectation_value_times()

    # size and shape of array depends on cost and gradient function
    initial_guess = np.zeros(2*times.size)

    # these are the max bounds in order to have convergence for espre =10**-6 maybe lb=-40,ub=40 but too heavy
    bounds = Bounds(lb=-20,ub=20)



    optimimised_result = minimize(
                            fun=qubit_reset_cost_function,
                            x0=initial_guess,
                            method='L-BFGS-B',
                            jac=qubit_reset_gradient,
                            bounds=bounds,
                            # args to be passed to cost & gradient functions
                            args=(pt),
                            options={'disp': True} # verbose result (optional)
                            )
    np.save(qsl_path + optimisation_name,optimimised_result.x)


if generate_dynamics:

    pt = load_process_tensor(
                    max_time,
                    dt,
                    dkmax,
                    esprel,
                    temp)
    # load *optimised* control parameters
    control_parameters = np.load(qsl_path + optimisation_name + '.npy')
    hamiltonian_t = get_ARP_hamiltonian(control_parameters, pt) # same as cost function
    system = tempo.AdjointTimeDependentSystem(hamiltonian_t)

    dynamics = pt.compute_dynamics_from_system(system,initial_state=initial_state)

    dynamics.export(dynamics_path + optimisation_name + '.dynamics',
                    overwrite=True)



if plot_results:
    control_parameters = np.load(qsl_path + optimisation_name + '.npy')
    control_parameters_initial_guess = initial_guess
    # load optimised dynamics
    dynamics = tempo.import_dynamics(dynamics_path+optimisation_name+'.dynamics')

    t, s_x = dynamics.expectations(tempo.operators.sigma('x'))
    t, s_y = dynamics.expectations(tempo.operators.sigma('y'))
    t, s_z = dynamics.expectations(tempo.operators.sigma('z'))

    times, states = dynamics.times, dynamics.states

    plt.plot(t, s_x.real, label=r'$<\sigma_{x}>$')
    plt.plot(t, s_y.real, label=r'$<\sigma_{y}>$')
    plt.plot(t, s_z.real, label=r'$<\sigma_{z}>$')
    plt.xlabel(r'$t(ns)$')
    plt.ylabel(r'$<\sigma_{i}>$')
    plt.title('Expectation Values')
    plt.grid()
    plt.legend()

    plt.show()


    plt.plot(times, states[:, 0, 0].real, label=r'$\rho_{00}$')
    plt.plot(times, states[:, 0, 1].real, label=r'$\rho_{01}$')
    plt.plot(times, states[:, 1, 0].real, label=r'$\rho_{10}$')
    plt.plot(times, states[:, 1, 1].real, label=r'$\rho_{11}$')
    
    plt.xlabel(r"$ t(ns)$")
    plt.ylabel(r"$ \rho_{ij}$")
    plt.grid()
    plt.title('Evolution of the density matrix elements dt=0.01, dkmax=80, epsrel=10**(-6)')
    plt.legend()

    plt.show()
    