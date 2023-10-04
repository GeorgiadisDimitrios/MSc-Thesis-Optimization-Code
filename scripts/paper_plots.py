import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rc
from scripts.quantum_speed_limit import get_dynamics_from_parameters

from path import figure_path, non_markovianity_path
import time_evolving_mpo as tempo

from quantum_speed_limit_optimisation import qsl_parameters,load_dynamics_from_parameter_class , get_qsl_parameter_class_from_dict,load_initial_condition_dynamics

from data_dictionary import quantum_speed_limit_dict

from slideshow_qsl_generator import get_closed_fidelity


def compare_two_maxtimes(time_1,
                        time_2,
                        parameters: qsl_parameters):

    assert isinstance(parameters,qsl_parameters)
    time_array = np.array([time_1,time_2])
    fig = plt.figure(figsize=(12,8))


    for i,time in enumerate(time_array):
        control_parameters,dynamics = get_dynamics_from_parameters(parameters.bound,time)


        times = dynamics.times[:-1]


        initial_guess = np.zeros(2 * times.size)

        initial_guess_sx = (np.pi / time) * np.ones(times.size) # constant pulse with area pi

        initial_guess[:times.size] = initial_guess_sx


        colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]



        ax = fig.add_subplot(2,2,2*i+1)


        ax.plot(times,control_parameters[:times.size],color=colours[0],label=r'$S_x$')
        ax.plot(times,control_parameters[times.size:],color=colours[1],label=r'$S_z$')
        ax.plot(times,initial_guess[:times.size],color=colours[0],linestyle='dashed',label=r'initial guess $S_x$')
        ax.plot(times,initial_guess[times.size:],color=colours[1],linestyle='dashed',label=r'initial guess $S_z$')
        ax.set_ylim(-parameters.bound-0.1,parameters.bound+0.1)

        ax.legend()

        #plt.ylabel(r'$\Gamma\quad [ps^{-1}]$')
        ax.set_xlabel('t(ps)')


        tempo.helpers.plot_bloch_sphere_for_slideshow(dynamics,fig=fig)

        t, s_x = dynamics.expectations(tempo.operators.sigma("x"), real=True)
        _, s_y = dynamics.expectations(tempo.operators.sigma("y"), real=True)
        _, s_z = dynamics.expectations(tempo.operators.sigma("z"), real=True)

        ax = fig.add_subplot(2,2,2*i+2)


        ax.plot(t,s_z,label=r"$\langle\sigma_z\rangle$")
        ax.plot(t,s_y,label=r"$\langle\sigma_y\rangle$")
        ax.plot(t,s_x,label=r"$\langle\sigma_x\rangle$")
        ax.set_xlabel(r"$t\,/\mathrm{ps}$")
        ax.set_xlim(0,time_2)
        ax.grid()
        ax.legend()
    plt.title('comparison_between_{}_and_{}_bound{}'.format(time_1,time_2,parameters.bound))
    plt.tight_layout()
    name = 'comparison_between_{}_and_{}_bound{}'.format(time_1,time_2,parameters.bound)
    name_replaced = name.replace('.','-')
    plt.savefig(figure_path + name_replaced + '.pdf')

def compare_bounds_and_unitary_dynamics(string_list):
    colours = ["C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
    plt.figure(figsize=(12,8))

    def get_closed_fidelity(time_array,bound):
        closed_fidelity = np.piecewise(time_array,[time_array < np.pi/bound,time_array >= np.pi/bound],[lambda t:(np.cos(bound*t / 2))**2,0])
        return closed_fidelity


    for count,i in enumerate(string_list):


        dynamics_list = []
        control_parameter_list = []

        qsl_dictionary  = quantum_speed_limit_dict[i]

        bound = qsl_dictionary['bound']
        max_time_array = qsl_dictionary['values']

        try:
            initial_string = qsl_dictionary['initial_state']
            target_string = qsl_dictionary['target_state']

            initial_state = tempo.operators.spin_dm(initial_string)
            target_state = tempo.operators.spin_dm(target_string)


        except:
            initial_state = tempo.operators.spin_dm('down')
            target_state = tempo.operators.spin_dm('up')

        parameters = qsl_parameters(bound,
                                i,
                                initial_state=initial_state,
                                target_state=target_state)


        for j in range(max_time_array.size):
            control_parameters,dynamics = load_dynamics_from_parameter_class(parameters,max_time_array[j])
            dynamics_list.append(dynamics)
            control_parameter_list.append(control_parameters)


        fidelity_array = np.zeros(max_time_array.size)
        for j in range(max_time_array.size):
            dynamics = dynamics_list[j]
            final_state = dynamics.states[-1,:,:]
            infidelity = 1 - np.trace(target_state @ final_state)
            fidelity_array[j] = infidelity

        if np.array_equal(initial_state,tempo.operators.spin_dm('down')):
            plt.plot(max_time_array,fidelity_array,color=colours[count],label='bound {}'.format(bound))
            plt.plot(max_time_array,fidelity_array,marker='x',color=colours[count])
            x_closed_fidelity = np.linspace(max_time_array[0],max_time_array[-1],500)
            closed_fidelity_array = get_closed_fidelity(x_closed_fidelity,bound)



            plt.plot(x_closed_fidelity,
                            closed_fidelity_array,
                            label = 'closed fidelity for {} bound'.format(bound),
                            color=colours[count],
                            linestyle='dashed')



        else:
            plt.plot(max_time_array,fidelity_array,color=colours[count-1],label='bound {} sx+ optimisation'.format(bound),linestyle='dashdot')
            plt.scatter(max_time_array,fidelity_array,marker='x',color=colours[count-1])



    plt.grid()
    plt.legend()
    plt.savefig('comparison_2.pdf')


def prl_single_fidelity(qsl_string):

    qsl_string = 'bound_1_sz_5_sx_V3'


    plt.rcParams['figure.constrained_layout.use'] = True
    parameters = get_qsl_parameter_class_from_dict(qsl_string)

    qsl_dictionary = quantum_speed_limit_dict[qsl_string]
    max_time_array = np.array(qsl_dictionary['values'])
    fidelities = np.zeros(max_time_array.size)
    initial_condition_infidelities = np.zeros(max_time_array.size)


    for i in range(max_time_array.size):

        dynamics = load_dynamics_from_parameter_class(parameters,max_time_array[i])[1]
        ic_dynamics = load_initial_condition_dynamics(parameters,max_time_array[i])[1]
        states = dynamics.states
        final_state = states[-1,:,:]
        target_state = parameters.target_state
        infidelity = 1 - np.trace(target_state @ final_state)

        fidelities[i] = infidelity

        ic_final_state = ic_dynamics.states[-1,:,:]
        ic_infidelity = 1 - np.trace(target_state @ ic_final_state)
        initial_condition_infidelities[i] = ic_infidelity



    cm = 1/2.54  # centimeters in inches
    width = 8.6
    aspect_ratio = 2/3
    height = width * aspect_ratio
    plt.figure(1,figsize=(width*cm,height*cm))
    plt.plot(max_time_array,fidelities,color='C0',label='Optimized')
    plt.plot(max_time_array,initial_condition_infidelities,color='C1',label='Unoptimized',linestyle='dashed')



    x_closed_fidelity = np.linspace(max_time_array[0],max_time_array[-1],100)
    closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
    plt.plot(x_closed_fidelity,closed_fidelity,linestyle='dashdot',color='green',label='Unitary')


    plt.xlabel('Pulse Duration [ps]')
    plt.ylabel('Infidelity')
    plt.legend()
    plt.show()

def prl_list(qsl_string_list:list,legend_list:list[str]):
    '''
    plots a few on the same plot
    '''
    plt.rcParams.update({'font.size': 8})


    cm = 1/2.54  # centimeters in inches
    width = 2*8.6
    aspect_ratio = 9/16
    height = width * aspect_ratio
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


    def plot_pane(_axs,_qsl_string_list,_legend_list)-> plt.axes:
            
        for k in range(len(_qsl_string_list)):

            qsl_string = _qsl_string_list[k]
            parameters = get_qsl_parameter_class_from_dict(qsl_string)

            qsl_dictionary = quantum_speed_limit_dict[qsl_string]
            max_time_array = np.array(qsl_dictionary['values'])
            fidelities = np.zeros(max_time_array.size)
            initial_condition_infidelities = np.zeros(max_time_array.size)


            for i in range(max_time_array.size):

                dynamics = load_dynamics_from_parameter_class(parameters,max_time_array[i])[1]
                ic_dynamics = load_initial_condition_dynamics(parameters,max_time_array[i])[1]
                states = dynamics.states
                final_state = states[-1,:,:]
                target_state = parameters.target_state
                infidelity = 1 - np.trace(target_state @ final_state)

                fidelities[i] = infidelity

                ic_final_state = ic_dynamics.states[-1,:,:]
                ic_infidelity = 1 - np.trace(target_state @ ic_final_state)
                initial_condition_infidelities[i] = ic_infidelity



            _axs.plot(max_time_array,fidelities,color=colours[2*k],label=_legend_list[k])

            _axs.plot(max_time_array,initial_condition_infidelities,color=colours[2*k+1],label='IC: '+_legend_list[k],linestyle='dashed')




        x_closed_fidelity = np.linspace(max_time_array[0],max_time_array[-1],100)
        closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
        _axs.axvline(np.pi,color='black',linestyle='dotted',linewidth=1.0)

        _axs.plot(x_closed_fidelity,closed_fidelity,linestyle='dashdot',color='green',label='B: Closed')



        _axs.set_ylabel('Infidelity')
        _axs.legend()

    if type(qsl_string_list[0]) is list:
        fig, axs = plt.subplots(nrows=1,ncols=len(qsl_string_list),constrained_layout=True,figsize=(width*cm,height*cm))
        for l in range(len(qsl_string_list)):
            plot_pane(axs[l],qsl_string_list[l],legend_list[l])
    else:
        fig, axs = plt.subplots(constrained_layout=True,figsize=(width*cm,height*cm))
        plot_pane(axs,qsl_string_list,legend_list)



    plt.show()

def lpb_and_single_fidelity(qsl_string):
    '''
    This is the one i've actually used
    '''
    plt.rcParams.update({'font.size': 8})
    qsl_string = 'bound_1_sz_5_sx_V3'


    parameters = get_qsl_parameter_class_from_dict(qsl_string)

    qsl_dictionary = quantum_speed_limit_dict[qsl_string]
    max_time_array = np.array(qsl_dictionary['values'])
    fidelities = np.zeros(max_time_array.size)
    initial_condition_infidelities = np.zeros(max_time_array.size)


    for i in range(max_time_array.size):

        dynamics = load_dynamics_from_parameter_class(parameters,max_time_array[i])[1]
        ic_dynamics = load_initial_condition_dynamics(parameters,max_time_array[i])[1]
        states = dynamics.states
        final_state = states[-1,:,:]
        target_state = parameters.target_state
        infidelity = 1 - np.trace(target_state @ final_state)

        fidelities[i] = infidelity

        ic_final_state = ic_dynamics.states[-1,:,:]
        ic_infidelity = 1 - np.trace(target_state @ ic_final_state)
        initial_condition_infidelities[i] = ic_infidelity


    cm = 1/2.54  # centimeters in inches
    width = 8.6
    aspect_ratio = 0.8
    height = width * aspect_ratio


    fig, axs = plt.subplots(ncols=1,nrows=2, constrained_layout=True,figsize=(width*cm,height*cm))

    insert_axes = inset_axes(axs[0],height=0.5,width=1.4)


    axs[0].plot(max_time_array,fidelities,color='C0',label='A: Open')
    axs[0].plot(max_time_array,initial_condition_infidelities,color='C1',label='B: Open',linestyle='dashed')




    x_closed_fidelity = np.linspace(max_time_array[0],max_time_array[-1],100)
    closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
    axs[0].plot(x_closed_fidelity,closed_fidelity,linestyle='dashdot',color='green',label='B: Closed')



    axs[0].set_ylabel('Infidelity')
    axs[0].text(2.0,0.7,'(a)')

    crop_index_optimisation = np.where(max_time_array==2.5)
    crop_index_closed_fidelity = np.where(x_closed_fidelity>2.5)[0]

    crop_index_optimisation = int(crop_index_optimisation[0])
    crop_index_closed_fidelity = int(crop_index_closed_fidelity[0])



    insert_axes.plot(max_time_array[crop_index_optimisation:],fidelities[crop_index_optimisation:],color='C0')
    insert_axes.plot(max_time_array[crop_index_optimisation:],initial_condition_infidelities[crop_index_optimisation:],color='C1',linestyle='dashed')
    insert_axes.plot(x_closed_fidelity[crop_index_closed_fidelity:],closed_fidelity[crop_index_closed_fidelity:],linestyle='dashdot',color='green')

    axs[0].legend(fontsize=7,loc='lower left',labelspacing=0.1, framealpha=1.0, handletextpad=0.1, borderpad=0.1,frameon=False)

    axs[0].axvline(np.pi,color='black',linestyle='dotted',linewidth=1.0)
    insert_axes.axvline(np.pi,color='black',linestyle='dotted',linewidth=1.0)

    lpb_measure_optimised = np.load(non_markovianity_path + qsl_string + 'lpb_measure_optimised.npy')
    lpb_measure_ic = np.load(non_markovianity_path + qsl_string + 'lpb_measure_ic.npy')


    axs[1].plot(max_time_array,lpb_measure_optimised,color='C0',label='A: Open')
    axs[1].plot(max_time_array,lpb_measure_optimised,color='C0',marker='x')

    axs[1].plot(max_time_array,lpb_measure_ic,color='C1',linestyle='dashed',label='B: Open')
    axs[1].text(2.0,0.06,'(b)')
    axs[1].set_ylabel('Non-Markovianity')
    axs[1].set_xlabel('Process Duration [ps]')
    axs[1].plot([],[],linestyle='dashdot',color='green',label='B: Closed')

    axs[1].axvline(np.pi,color='black',linestyle='dotted',linewidth=1.0)


    plt.show()

def prl_multi_panel_fidelity(qsl_list):
    '''
    multi panel infidelity plots for PRL journal width

    qsl_list_format = [[bound_1_string1,bound_1string2...,[bound_2_string1],[bound_3_string_1,bound_3_string2...].....]
    '''

    def plot_single_pane(qsl_string,ax):



        parameters = get_qsl_parameter_class_from_dict(qsl_string)

        qsl_dictionary = quantum_speed_limit_dict[qsl_string]
        max_time_array = np.array(qsl_dictionary['values'])
        fidelities = np.zeros(max_time_array.size)



        for i in range(max_time_array.size):

            dynamics = load_dynamics_from_parameter_class(parameters,max_time_array[i])[1]
            states = dynamics.states
            final_state = states[-1,:,:]
            target_state = parameters.target_state
            infidelity = 1 - np.trace(target_state @ final_state)


            fidelities[i] = infidelity


        if np.array_equal(parameters.initial_state,tempo.operators.spin_dm('x-')):
            ax.plot(max_time_array,fidelities,color='C0')
        else:
            ax.plot(max_time_array,fidelities,color='black')
        ax.plot(max_time_array,fidelities,'C1x')

        return ax


    cm = 1/2.54  # centimeters in inches
    width = 8.6
    aspect_ratio = 3/2
    height = width * aspect_ratio


    fig, axs = plt.subplots(ncols=1,nrows=len(qsl_list),constrained_layout=True,figsize=(width*cm,height*cm))


    for i in range(len(qsl_list)):
        if type(qsl_list[i]) is list:
            if len(qsl_list) != 1:
                current_axs = axs[1]
            else:
                current_axs = axs
            for j in range(len(qsl_list[i])):
                current_axs = plot_single_pane(qsl_list[i][j],current_axs)





            parameters = get_qsl_parameter_class_from_dict(qsl_list[i][0]) # take first qsl string and hope that it covers everything

            qsl_dictionary = quantum_speed_limit_dict[qsl_list[i][0]]
            max_time_array = np.array(qsl_dictionary['values'])
            x_closed_fidelity = np.linspace(max_time_array,max_time_array,100)

            closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
            current_axs.plot(x_closed_fidelity,closed_fidelity,color='C3',linestyle='dashdot')
        else:
            axs[i] = plot_single_pane(qsl_list[i],axs[i])

            parameters = get_qsl_parameter_class_from_dict(qsl_list[i]) # take first qsl string and hope that it covers everything

            qsl_dictionary = quantum_speed_limit_dict[qsl_list[i]]
            max_time_array = np.array(qsl_dictionary['values'])

            x_closed_fidelity = np.linspace(max_time_array,max_time_array,100)

            closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
            axs[i].plot(x_closed_fidelity,closed_fidelity ,color='C3',linestyle='dashdot')
    fig.supxlabel('Pulse Duration [ps]')
    fig.supylabel('infidelity')
    plt.show()


prl_list([['bound_1_sz_5_sx_V3','pure_dephasing_v4'],['bound_1_sz_5_sx_V3','pure_dephasing_ic_guess','pure_dephasing_ic_guess_2']],[['A: Open',r'$\gamma=0.015$'],['A: Open','linear IC','non-linear IC']])
prl_list(['closed_system_optimisation','closed_system_optimisation_better_IC'],['hamiltonian linear','hamiltonian non-linear'])