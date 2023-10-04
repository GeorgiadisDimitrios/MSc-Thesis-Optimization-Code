import numpy as np

import matplotlib.pyplot as plt
from scripts.cost_function import get_ARP_hamiltonian

from quantum_speed_limit_optimisation import qsl_parameters, load_process_tensor, get_qsl_parameter_class_from_dict, load_initial_condition_dynamics

import time_evolving_mpo as tempo

from data_dictionary import optimisation_result_dict, quantum_speed_limit_dict

from path import qsl_path

from slideshow_qsl_generator import get_phonon_rates

plot_hamiltonian = False
check_hamiltonian = False
plot_dynamics = True
save_dynamics = False
plot_bloch_sphere = True
plot_bloch_vector_magnitude = False
plot_phonon_rates = False

plot_paper_dynamics_and_hamiltonian = True
result_type = 'qsl_optimisation_parameters'


if result_type == 'data_dictionary_string':

    optimisation_result_string = 'QSL_optimisation_0-5ps'


    optimisation_dict = optimisation_result_dict[optimisation_result_string]
    control_parameters = np.load(optimisation_dict['control_parameters'])

    pt = tempo.import_adjoint_process_tensor(optimisation_dict['process_tensor'],'adjoint') # original pt



if result_type == 'qsl_optimisation_parameters':



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    qsl_string = 'bound_1_sz_5_sx_V3'
    max_time = 5.0 # should be float
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    qsl_dictionary = quantum_speed_limit_dict[qsl_string]


    bound = qsl_dictionary['bound']
    loop_array = qsl_dictionary['values']

    parameters = get_qsl_parameter_class_from_dict(qsl_string)

    initial_state = parameters.initial_state

    assert isinstance(max_time,float)

    cp_filename = parameters.name + '_time_{}'.format(max_time) # control parameter filename

    cp_filename_replaced = cp_filename.replace('.','-')
    control_parameters = parameters.get_initial_guess_array(max_time)

    pt = load_process_tensor(max_time,parameters)


if result_type == 'constant_field':

    optimisation_result_string = 'take_two'



    pt = tempo.import_adjoint_process_tensor(qsl_path + 'optimisation_dt0-05dkmax60esprel1e-08temp5alpha0-252maxtime6-4.processTensor','adjoint') # original pt
    control_parameters = np.array([5,0])
    initial_state = tempo.operators.spin_dm('x-')




assert isinstance(pt,tempo.SimpleAdjointTensor), 'must be an adjoint tensor'


# shorter timestep comparison pt
comparison_pt = None

if comparison_pt is not None:
    comparison_pt_times = comparison_pt.get_expectation_value_times() #  needed to plot hamiltonian
    print('comparison_pt')
    print(comparison_pt_times)
    print(comparison_pt_times.size)

lower_bound = 0.0
upper_bound = 5.0

times = pt.get_expectation_value_times()
if comparison_pt is not None:
    print('smaller_pt')
    print(times)
    print(times.size)

initial_guess = parameters.get_initial_guess_array(max_time)

if result_type == 'constant_field':
    hx = control_parameters[0] * np.ones(times.size)
    hz = control_parameters[1] * np.ones(times.size)
    control_parameters = np.concatenate((hx,hz))
    hamiltonian_t = get_ARP_hamiltonian(control_parameters,pt)
else:
    hamiltonian_t = get_ARP_hamiltonian(control_parameters,pt)

colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


if plot_hamiltonian:
    plt.figure(1)
    if comparison_pt is not None:
        plt.plot(times,control_parameters[:times.size],color=colours[0],label=r'$S_x$')
        plt.plot(times,control_parameters[times.size:],color=colours[1],label=r'$S_z$')
        plt.plot(times,initial_guess[:times.size],color=colours[0],linestyle='dashed',label=r'initial guess $S_x$')
        plt.plot(times,initial_guess[times.size:],color=colours[1],linestyle='dashed',label=r'initial guess $S_z$')

    else:
        plt.plot(times,control_parameters[:times.size],color=colours[0],label=r'$S_x$')
        plt.plot(times,control_parameters[times.size:],color=colours[1],label=r'$S_z$')
        plt.plot(times,initial_guess[:times.size],color=colours[0],linestyle='dashed',label=r'initial guess $S_x$')
        plt.plot(times,initial_guess[times.size:],color=colours[1],linestyle='dashed',label=r'initial guess $S_z$')

    plt.legend()

    plt.xlabel('t(ps)')
    plt.grid()
    if result_type == 'data_dictionary_string':
        plt.savefig('optimised_hamiltonian_{}.pdf'.format(optimisation_result_string))

if check_hamiltonian:
    system = tempo.AdjointTimeDependentSystem(hamiltonian_t)

    expval_times = pt.get_derivative_times()
    hamiltonian_values = np.zeros((expval_times.size,system.dimension,system.dimension))
    for i in range(expval_times.size):
        hamiltonian_values[i,:,:] = hamiltonian_t(expval_times[i])


    plt.figure(4)
    plt.plot(expval_times,hamiltonian_values[:,0,0],label='z')
    plt.plot(expval_times,hamiltonian_values[:,0,1],label='x')
    plt.xlabel('t')
    plt.ylabel(r'$\sigma_i$')
    plt.legend()



if plot_dynamics or plot_bloch_sphere or save_dynamics\
     or check_hamiltonian or plot_bloch_vector_magnitude:
    system = tempo.AdjointTimeDependentSystem(hamiltonian_t)

    if comparison_pt is None:
        dynamics = pt.compute_dynamics_from_system(
                system=system,
                initial_state=initial_state)
    else:
        dynamics = comparison_pt.compute_dynamics_from_system(
                system=system,
                initial_state=initial_state)


if plot_paper_dynamics_and_hamiltonian:
    '''
    adding initial guess bloch vector to dynamics
    '''

    ic_dynamics = load_initial_condition_dynamics(parameters,max_time)[1]



if plot_bloch_sphere:
    plt.figure(3)
    if result_type == 'data_dictionary_string':
        tempo.helpers.plot_bloch_sphere(dynamics,update_rc=False,save=True,filename='bloch_sphere_{}.pdf'.format(optimisation_result_string))
    else:
        tempo.helpers.plot_bloch_sphere(dynamics,update_rc=False)


if save_dynamics:
    dynamics.export('dynamics_1.dynamics',overwrite=True)

if plot_dynamics:
    density_matrix = dynamics.states
    final_density_matrix = density_matrix[-1,:,:]
    fidelity = 1 - np.matmul(final_density_matrix,tempo.operators.spin_dm("z+")).trace()
    print('computed infidelity is {}'.format(fidelity))

    t, s_x = dynamics.expectations(tempo.operators.sigma("x"), real=True)
    _, s_y = dynamics.expectations(tempo.operators.sigma("y"), real=True)
    _, s_z = dynamics.expectations(tempo.operators.sigma("z"), real=True)

    if plot_phonon_rates:


        fig4,ax4 = plt.subplots()

        ax4.plot(t,s_z,label=r"$\langle\sigma_z\rangle$")
        ax4.plot(t,s_y,label=r"$\langle\sigma_y\rangle$")
        ax4.plot(t,s_x,label=r"$\langle\sigma_x\rangle$")
        ax4.set_xlabel(r"$t\,/\mathrm{ps}$")
        ax4.grid()

        ax4_2 = ax4.twinx()
        phonon_absorbtion,phonon_emission = get_phonon_rates(control_parameters[:times.size],control_parameters[times.size:],5)

        ax4_2.plot(t[:-1],phonon_absorbtion,color='black',label = 'phonon absorbtion',linestyle = 'dashed')
        ax4_2.plot(t[:-1],phonon_emission,color='black',label = 'phonon emission')

        fig4.legend()
        if result_type == 'data_dictionary_string':
            fig4.savefig('optimised_dynamics_{}.pdf'.format(optimisation_result_string))
    else:
        plt.figure(5)

        plt.plot(t,s_z,label=r"$\langle\sigma_z\rangle$")
        plt.plot(t,s_y,label=r"$\langle\sigma_y\rangle$")
        plt.plot(t,s_x,label=r"$\langle\sigma_x\rangle$")
        plt.xlabel(r"$t\,/\mathrm{ps}$")
        plt.grid()



if plot_bloch_vector_magnitude:

    t, s_x = dynamics.expectations(tempo.operators.sigma("x"), real=True)
    _, s_y = dynamics.expectations(tempo.operators.sigma("y"), real=True)
    _, s_z = dynamics.expectations(tempo.operators.sigma("z"), real=True)

    bloch_vector_magnitude = np.sqrt(s_x**2 + s_y**2 + s_z**2)

    plt.figure(6)
    plt.plot(t,bloch_vector_magnitude)
    plt.xlabel('t')
    plt.ylabel(r'$\sqrt{\langle\sigma_x\rangle+\langle\sigma_y\rangle+\langle\sigma_z\rangle}$')
    plt.grid()
    plt.tight_layout()
    if result_type == 'data_dictionary_string':
        plt.savefig('bloch_vector_magnitude_{}'.format(optimisation_result_string))

if plot_paper_dynamics_and_hamiltonian:
    plt.rcParams.update({'font.size': 8})


    cm = 1/2.54  # centimeters in inches
    width = 8.6
    aspect_ratio = 0.8
    height = width * aspect_ratio


    fig, axs = plt.subplots(ncols=1,nrows=2, constrained_layout=True,figsize=(width*cm,height*cm))

    x_colour = colours[2]
    y_colour = colours[0]
    z_colour = colours[1]
    bloch_colour = 'black'

    axs[0].plot(times,control_parameters[:times.size],color=x_colour,label=r'$h_x$')
    axs[0].plot(times,control_parameters[times.size:],color=z_colour,label=r'$h_z$')

    axs[0].text(3.2,-5,r'$h_x$',color=x_colour)
    axs[0].text(3.2,2,r'$h_z$',color=z_colour)



    if parameters.bound_sx is None:
        bound_sx = parameters.bound
    else:
        bound_sx = parameters.bound_sx

    bound_sz = parameters.bound

    axs[0].set_ylabel(r'$h_{\bullet}\ [ps^{-1}]$')
    axs[0].text(3.5,4,'(a)')



    axs[1].plot(t,s_x,label=r"$\langle\sigma_x\rangle$",color=x_colour,linestyle='solid')
    axs[1].plot(t,s_y,label=r"$\langle\sigma_y\rangle$",color=y_colour,linestyle=(0, (3, 1, 1, 1, 1, 1)))
    axs[1].plot(t,s_z,label=r"$\langle\sigma_z\rangle$",color=z_colour,linestyle='dashed')

    axs[1].text(2.5,0.75,r"$\langle\sigma_x\rangle$",color=x_colour)
    axs[1].text(2.5,-1,r"$\langle\sigma_y\rangle$",color=y_colour)
    axs[1].text(2.5,-0.4,r"$\langle\sigma_z\rangle$",color=z_colour)


    axs[1].text(3.5,-0.9,'(b)')
    axs[1].set_ylabel(r'$\langle\sigma_{\bullet}\rangle$')
    axs[1].set_xlabel('Time [ps]')

    bloch_axs = axs[1].twinx()

    ic_t,ic_sigma_x = ic_dynamics.expectations(tempo.operators.sigma('x'),real=True)
    ic_t,ic_sigma_y = ic_dynamics.expectations(tempo.operators.sigma('y'),real=True)
    ic_t,ic_sigma_z = ic_dynamics.expectations(tempo.operators.sigma('z'),real=True)

    ic_bloch_vector_magnitude = np.sqrt(ic_sigma_x**2 + ic_sigma_y**2 + ic_sigma_z**2)

    bloch_vector_magnitude = np.sqrt(s_x**2 + s_y**2 + s_z**2)

    linewidth = 1.0

    bloch_axs.plot(t,bloch_vector_magnitude,color=bloch_colour,linewidth=linewidth)
    bloch_axs.plot(t,ic_bloch_vector_magnitude,color=bloch_colour,linestyle='dashed',linewidth=linewidth)
    bloch_axs.set_ylim(0.7,1.02)

    bloch_axs.set_ylabel(r'$|\langle\mathbf{\sigma}\rangle|$')







plt.show()
