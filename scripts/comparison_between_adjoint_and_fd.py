'''
Compare adjoint method and brute force optimisation


'''

import numpy as np
import matplotlib.pyplot as plt

import time_evolving_mpo as tempo

from finite_difference_routines import compute_finite_difference_brute
from cost_function import ARP_protocol_gradient_function, ARP_gradient_brute

from quantum_speed_limit_optimisation import get_qsl_parameter_class_from_dict, load_process_tensor
from data_dictionary import quantum_speed_limit_dict


def legacy_comparison():
    '''
    the OG function for the legacy comparison
    '''


    # which plots dya want?
    plot_dynamics = False
    plot_brute = True


    # import the process tensor as an adjoint pt and set up the system
    pt = tempo.import_adjoint_process_tensor('details_pt_tempo_dt0p02.processTensor','adjoint')

    delta=0.0

    initial_state=tempo.operators.spin_dm("z-")

    def gaussian_shape(t, area = 1.0, tau = 1.0, t_0 = 0.0):
        return area/(tau*np.sqrt(np.pi)) * np.exp(-(t-t_0)**2/(tau**2))

    def hamiltonian_t(t, delta=delta):
            return delta/2.0 * tempo.operators.sigma("z") + gaussian_shape(t-2.5, area = np.pi/2.0, tau = 0.245)/2.0 * tempo.operators.sigma("x")

    system = tempo.AdjointTimeDependentSystem(hamiltonian_t)


    # compute the dynamics using normal PT-TEMPO code for comparison
    dynamics = pt.compute_dynamics_from_system(
            system=system,
            initial_state=initial_state)
    t, s_x = dynamics.expectations(tempo.operators.sigma("x"), real=True)
    np.save('fd_t', t)
    _, s_y = dynamics.expectations(tempo.operators.sigma("y"), real=True)
    _, s_z = dynamics.expectations(tempo.operators.sigma("z"), real=True)
    s_xy = np.sqrt(s_x**2 + s_y**2)

    # def allproptimes(times,dt):
    #     return np.array([[t+dt/4.0,t+dt*3.0/4.0] for t in times]).flatten()
    # times = 0.0 + np.arange(len(pt.forwards)/2)*pt.dt
    # preposttimes=allproptimes(times,pt.dt)

    # np.save('fd_times',preposttimes)

    Omega_t = gaussian_shape(t-2.5, area = np.pi/2.0, tau = 0.245)

    if plot_dynamics:
        plt.rc('font',size=18)
        plt.figure(1,figsize=(10,6))
        plt.fill_between(t, Omega_t/4.0, 0,
                        facecolor="orange", # The fill color
                        color='blue',       # The outline color
                        alpha=0.2,label=r"Driving pulse")

        plt.plot(t,s_z,label=r"$\langle\sigma_z\rangle$")
        plt.plot(t,s_y,label=r"$\langle\sigma_y\rangle$")
        plt.plot(t,s_x,label=r"$\langle\sigma_x\rangle$")
        plt.xlabel(r"$t\,/\mathrm{ps}$")
        #plt.ylabel(r"$\mathrm{ps}^{-1}$")
        plt.legend()
        plt.savefig("fig1.pdf")

    if plot_brute:
        target_state = tempo.operators.spin_dm("y+")
        y_deriv_x = compute_finite_difference_brute(pt,hamiltonian_t,initial_state,target_state,t,tempo.operators.sigma('x'))
        np.save('y_deriv_x',y_deriv_x)
        print('done sx')
        y_deriv_y = compute_finite_difference_brute(pt,hamiltonian_t,initial_state,target_state,t,tempo.operators.sigma('y'))
        np.save('y_deriv_y',y_deriv_y)
        print('done sy')
        y_deriv_z = compute_finite_difference_brute(pt,hamiltonian_t,initial_state,target_state,t,tempo.operators.sigma('z'))
        np.save('y_deriv_z',y_deriv_z)
        print('done sz')




    if plot_dynamics or plot_brute:
        plt.show()




def comparison_from_qsl_str(qsl_string,index: int=-1):
    '''
    takes a qsl string and sees if the first optimisation iteration gradient is the same
    as that found by finite difference#

    index -> the index for the array of max_times
    '''

    compute_fd_derivs = False

    parameters = get_qsl_parameter_class_from_dict(qsl_string)

    qsl_dict = quantum_speed_limit_dict[qsl_string]

    loop_array = qsl_dict['values']

    print('using a max time of')#
    print(str(loop_array[index])+ ' ps')

    pt = load_process_tensor(loop_array[index],parameters)

    control_parameters = parameters.get_initial_guess_array(loop_array[index])

    def compute_fd_gradient():
        derivs = ARP_gradient_brute(control_parameters,pt,parameters.initial_state,parameters.target_state)
        np.save('derivs',derivs)
    
    if compute_fd_derivs:
        compute_fd_gradient()

    fd_derivs = np.load('derivs.npy')

    adj_derivs = ARP_protocol_gradient_function(control_parameters,pt,parameters.initial_state,parameters.target_state)

    difference = fd_derivs-adj_derivs

    plt.figure(1)
    plt.plot(difference)
    plt.title('difference')

    plt.figure(2)

    plt.plot(fd_derivs,label='fd_derivs')
    plt.plot(adj_derivs,label='adj_derivs')

    plt.legend()

    plt.show()

comparison_from_qsl_str('bound_1')



    

    



    



    