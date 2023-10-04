'''
All of the cost functions and helper functions
'''

import matplotlib.pyplot as plt


import numpy as np
from scipy.interpolate import interp1d
import time_evolving_mpo as tempo


from finite_difference_routines import time_dependent_finite_difference

def qubit_reset_cost_function(control_parameters,
                        pt,
                        initial_state = tempo.operators.spin_dm("mixed"),
                        target_state = tempo.operators.spin_dm("z-")):


    hamiltonian_t = get_ARP_hamiltonian(control_parameters,pt)

    system = tempo.AdjointTimeDependentSystem(hamiltonian_t)
    final_state = pt.compute_final_state_from_system(system,initial_state=initial_state)
    fidelity = 1 - np.matmul(final_state,target_state).trace() # infidelity
    print('infidelity: {}'.format(fidelity.real))
    # delete when updated for transmon
    #raise NotImplementedError

    return fidelity.real



def qubit_reset_gradient(control_parameters, # first half controls Sx, second, Sz
                        pt,
                        initial_state = tempo.operators.spin_dm("mixed"),
                        target_state = tempo.operators.spin_dm("z-")):

    expval_time = pt.get_expectation_value_times()

    hamiltonian_t = get_ARP_hamiltonian(control_parameters,pt)

    system = tempo.AdjointTimeDependentSystem(hamiltonian_t)

    target_derivs = pt.compute_derivatives_from_system(system,
                        initial_state=initial_state,
                        target_state = target_state)

    propagator_timesteps = pt.get_derivative_times()


    omega_derivs = time_dependent_finite_difference(pt,
                                system,expval_time,
                                0.5 * tempo.operators.sigma('x'),
                                10**(-6))
    delta_derivs = time_dependent_finite_difference(pt,
                                system,expval_time,
                                0.5 * tempo.operators.sigma('z'),
                                10**(-6))

    total_derivs = np.concatenate((omega_derivs,delta_derivs))

    total_derivs_fortran = np.asfortranarray(total_derivs.real)
    return total_derivs_fortran


def get_ARP_hamiltonian(control_parameters,pt):
    assert isinstance(pt,tempo.SimpleAdjointTensor), ' pt must be a simple adjoint tensor'
    delta_t_short = control_parameters[:len(pt)] # first half is sigma x
    omega_t_short = control_parameters[len(pt):] # second half is sigma z

    # interp doesn't extrapolate beyond the last point so any time after t-dt will be out of bounds
    # need to duplicate the final point so we create the very last 'pixel
    expval_times = pt.get_expectation_value_times()
    #expval_times plus one timestep
    expval_times_p1 = np.concatenate((expval_times,np.array([pt.dt * len(pt)])))
    # duplicate last element so any time between t_f-dt and t_f falls within this 'pixel'
    # otherwise scipy interp1d doesn't like extrapolating so calls it out of bounds
    omega_t_p1 = np.concatenate((omega_t_short,np.array([omega_t_short[-1]])))
    delta_t_p1 = np.concatenate((delta_t_short,np.array([delta_t_short[-1]])))

    omega_t_interp = interp1d(expval_times_p1,omega_t_p1,kind='zero')
    delta_t_interp = interp1d(expval_times_p1,delta_t_p1,kind='zero')

    def hamiltonian_t(t):
        omega_t = omega_t_interp(t)
        delta_t = delta_t_interp(t)

        omega_sz = 0.5 * tempo.operators.sigma('z') * omega_t
        delta_sx = 0.5 * tempo.operators.sigma('x') * delta_t
        hamiltonian = omega_sz + delta_sx

        return hamiltonian
    
    
    times = np.linspace(0,5,100)
    omega_plot = [omega_t_interp(t) for t in times]
    plt.plot(times,omega_plot)
    plt.xlabel('t(ns)')
    plt.ylabel('control fields (GHz)')
    plt.show()
    return hamiltonian_t
