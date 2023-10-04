

import numpy as np

import time_evolving_mpo as tempo
from generate_adjoint_pt import generate_pt_QSL_QD
from tqdm import tqdm

from scipy.optimize import minimize, Bounds

from math import isclose

from cost_function import ARP_protocol_cost_function, ARP_protocol_gradient_function, get_ARP_dynamics, three_field_cost_function, three_field_gradient_function, get_three_field_dynamics
from quantum_speed_limit import get_dynamics_from_parameters, qsl


import matplotlib.pyplot as plt

from path import qsl_path, dynamics_path

from data_dictionary import quantum_speed_limit_dict

class qsl_parameters:
    def __init__(self,
                bound:float,
                name:str,
                dt:float = 0.05,
                dkmax:int = 60,
                esprel:float = 10**(-7),
                temperature = 5,
                alpha = None,
                figname = None,
                type = 'two_field',
                bound_sx = None,
                pt_name = None,
                initial_condition = None,
                pt_tempo_parameters = None,
                initial_state:np.ndarray = tempo.operators.spin_dm('down'),
                target_state:np.ndarray = tempo.operators.spin_dm('up'),
                dissipator_rate = None

                ) -> None:

        self.bound = bound
        self.bound_sx = bound_sx
        self.dt = dt
        self.dkmax = dkmax
        self.esprel = esprel
        self.temperature = temperature
        self.alpha = alpha
        self.figname = figname

        lower_bound = -1*bound
        upper_bound = bound
        self.bounds_instance = Bounds(lb=lower_bound,ub=upper_bound)

        self.three_field = False
        self.objective_landscape = False

        if type == 'three_field':
            self.three_field = True
        elif type == 'objective_landscape':
            self.objective_landscape = True

        self.initial_state = initial_state
        self.target_state = target_state
        self.name = name
        self.pt_name = pt_name
        self.pt_tempo_parameters = pt_tempo_parameters
        self.initial_condition_tuple = initial_condition
        self.dissipator_rate = dissipator_rate


    def update_bound_instance(self,
                            lower_bound:np.ndarray,
                            upper_bound:np.ndarray):
        """
        refreshes the bounds instance needed for scipy.minimize
        """
        _bounds_instance = Bounds(lb=lower_bound,ub=upper_bound)
        self.bounds_instance = _bounds_instance

        return self.bounds_instance

    def get_times(self,max_time): 
        number_of_MPOs = max_time / self.dt

        assert isclose(number_of_MPOs%1,0) or isclose(number_of_MPOs%1,1),\
        'max time is not an integer number of timesteps' # does the check


        number_of_MPOs_floor = int(max_time/self.dt)
        if self.dissipator_rate is not None:
            number_of_MPOs_floor = int(np.round(number_of_MPOs,decimals=0))


        expval_times = np.arange(0,number_of_MPOs_floor) * self.dt
        return expval_times

    def get_initial_guess_array(self,max_time) -> np.ndarray:
        expval_times = self.get_times(max_time)
        if isinstance(self.initial_condition_tuple,np.ndarray):

            if self.three_field == False:
                initial_guess = np.zeros(expval_times.size*2)
                initial_guess[:expval_times.size] = self.initial_condition_tuple[0]
                initial_guess[expval_times.size:] = self.initial_condition_tuple[1]
                return initial_guess
            else:
                initial_guess = np.zeros(expval_times.size*3)
                initial_guess[:expval_times.size] = self.initial_condition_tuple[0]
                initial_guess[expval_times.size:expval_times.size*2] = self.initial_condition_tuple[1]
                initial_guess[expval_times.size*2:expval_times.size*3] = self.initial_condition_tuple[2]

                return initial_guess

        else:
            if self.initial_condition_tuple is None:
                if np.array_equal(self.initial_state,tempo.operators.spin_dm('x-'))\
                    or np.array_equal(self.initial_state,tempo.operators.spin_dm('x+')):
                    self.initial_condition_tuple = 'pi_hz'
                else:
                    self.initial_condition_tuple = 'pi_hx'

            if self.three_field == True:
                raise NotImplementedError
            initial_guess = np.zeros(2*expval_times.size)
            initial_guess_nonzero_part = (np.pi / max_time)
            if self.initial_condition_tuple == 'pi_hx':
                if self.bound_sx is None:
                    if initial_guess_nonzero_part > self.bound:
                        initial_guess_nonzero_part = self.bound
                else:
                    if initial_guess_nonzero_part > self.bound_sx:
                        initial_guess_nonzero_part = self.bound_sx
                initial_guess[:expval_times.size] = initial_guess_nonzero_part
            elif self.initial_condition_tuple == 'pi_hz':
                if initial_guess_nonzero_part > self.bound:
                    initial_guess_nonzero_part = self.bound
                initial_guess[expval_times.size:] = initial_guess_nonzero_part
            elif self.initial_condition_tuple == 'markov_random':
                initial_guess[:expval_times.size] = 0.1
                initial_guess[expval_times.size:] = np.sin(np.linspace(0,5,expval_times.size)
                        *2*np.pi/5)*0.1

            else:
                raise KeyError('Unrecognised string')
            return initial_guess


def get_qsl_parameter_class_from_dict(qsl_dict_string:str)->qsl_parameters:


    qsl_dictionary  = quantum_speed_limit_dict[qsl_dict_string]

    bound = qsl_dictionary['bound']
    loop_array = qsl_dictionary['values']


    try:
        initial_string = qsl_dictionary['initial_state']
        target_string = qsl_dictionary['target_state']

        initial_state = tempo.operators.spin_dm(initial_string)
        target_state = tempo.operators.spin_dm(target_string)

    except KeyError:
        initial_state = tempo.operators.spin_dm('down')
        target_state = tempo.operators.spin_dm('up')


    try:
        bound_sx = qsl_dictionary['bound_sx']

    except KeyError:
        bound_sx = None

    try:
        pt_name = qsl_dictionary['pt_name']
    except KeyError:
        pt_name = None

    try:
        pt_tempo_parameters = qsl_dictionary['pt_parameters']
        dt = pt_tempo_parameters['dt']
        dkmax = pt_tempo_parameters['dkmax']
        esprel = pt_tempo_parameters['esprel']
        temp = pt_tempo_parameters['temp']
        try:
            alpha = pt_tempo_parameters['alpha']
        except KeyError:
            alpha = None

    except KeyError:
        dt=0.05
        dkmax = 60
        esprel=1e-07
        temp=5
        alpha = None


    try:
        ic_string = qsl_dictionary['initial_condition']
    except KeyError:
        ic_string = None


    try:
        dissipator_rate = qsl_dictionary['dissipator_rate']
    except KeyError:
        dissipator_rate = None



    parameters_instance = qsl_parameters(bound,
                        type=qsl_dictionary['type'],
                        name=qsl_dict_string,
                        initial_state=initial_state,
                        target_state=target_state,
                        bound_sx=bound_sx,
                        pt_name=pt_name,
                        dt=dt,
                        dkmax=dkmax,
                        esprel=esprel,
                        temperature=temp,
                        alpha=alpha,
                        initial_condition=ic_string,
                        dissipator_rate=dissipator_rate

                        )

    return parameters_instance



def qsl_optimisation(parameters,
                    max_time_array,
                    plot_graphs = True,
                    run_optimisation = False,
                    compute_dyn = False,
                    generate_PTs = False,
                    ic_compute_dyn = False
                    ):

    assert isinstance(parameters,qsl_parameters)

    if generate_PTs:
        generate_process_tensors(max_time_array,parameters)

    if run_optimisation:
        print('===================================')
        print('doing the ' + parameters.name + ' optimisation')
        print('===================================')
        print('running the optimisation code......')
        print('attempting to load process tensors:')

        success = True

        for i in range(max_time_array.size):
            try:
                load_process_tensor(max_time_array[i],parameters)
            except FileNotFoundError:
                success = False
                print('cannot load process tensor for max_time = {} ps'.format(max_time_array[i]))
                print('generating_process_tensor')
                generate_pt_QSL_QD(dt=parameters.dt,dkmax=parameters.dkmax,esprel=parameters.esprel,max_time=max_time_array[i],_temperature=parameters.temperature, plot_correlations=False)

        if success:
            print('success')

        do_optimisation(max_time_array,parameters)



    if compute_dyn or ic_compute_dyn:
        print('generating the dynamics............')
        print('attempting to load process tensors:')

        success = True

        for i in range(max_time_array.size):
            try:
                load_process_tensor(max_time_array[i],parameters)
            except OSError:
                success = False
                print('cannot load process tensor for max_time = {} ps'.format(max_time_array[i]))
                print('generating_process_tensor')
                generate_pt_QSL_QD(dt=parameters.dt,dkmax=parameters.dkmax,esprel=parameters.esprel,max_time=max_time_array[i],_temperature=parameters.temperature, plot_correlations=False)

        if success:
            print('success')
        if compute_dyn:
            compute_dynamics(max_time_array,parameters)
        if ic_compute_dyn:
            compute_initial_condition_dynamics(max_time_array,parameters)


    if plot_graphs:
        compare_results_with_QSL(max_time_array,parameters)


def generate_process_tensors(max_time_array,parameters):
    assert isinstance(parameters,qsl_parameters)
    for i in tqdm(range(max_time_array.size)):
        if parameters.alpha is None:
            generate_pt_QSL_QD(dt=parameters.dt,dkmax=parameters.dkmax,esprel=parameters.esprel,max_time=max_time_array[i],_temperature=parameters.temperature, plot_correlations=False)
        else:
            generate_pt_QSL_QD(dt=parameters.dt,dkmax=parameters.dkmax,esprel=parameters.esprel,max_time=max_time_array[i],_temperature=parameters.temperature,alpha=parameters.alpha, plot_correlations=False)



def _do_optimisation(max_time,parameters):
    pt =  load_process_tensor(max_time,parameters)

    times = pt.get_expectation_value_times()

    initial_guess = parameters.get_initial_guess_array(max_time)

    _bounds_instance = parameters.bounds_instance

    if parameters.bound_sx is not None:
        upper_bound_array = np.zeros(2 * times.size)
        upper_bound_array[:times.size] = parameters.bound_sx
        upper_bound_array[times.size:] = parameters.bound

        lower_bound_array = -1 * upper_bound_array

        _bounds_instance = parameters.update_bound_instance(lower_bound=lower_bound_array, upper_bound=upper_bound_array)

    optimised_result = minimize(ARP_protocol_cost_function,
                        initial_guess,method='L-BFGS-B',
                        jac=ARP_protocol_gradient_function,
                        bounds=_bounds_instance,
                        args=(pt,parameters.initial_state,parameters.target_state),
                        options={'disp': True,'gtol': 7e-05}
                        )


    return optimised_result

def _do_optimisation_three_field(max_time,parameters):

    pt = load_process_tensor(max_time,parameters)
    times = pt.get_expectation_value_times()

    assert len(pt) == times.size
    initial_guess = None

    if initial_guess is None:
        initial_guess = np.zeros(3 * times.size)
        initial_guess_sx = (np.pi / max_time) * np.ones(times.size)

        initial_guess[:times.size] = initial_guess_sx
        optimised_result = minimize(three_field_cost_function,initial_guess,method='L-BFGS-B',jac=three_field_gradient_function, bounds=parameters.bounds_instance, args=(pt,parameters.initial_state,parameters.target_state))

    else:
        assert 3 * len(pt) == initial_guess.size
    optimised_result = minimize(three_field_cost_function,
                            initial_guess,
                            method='L-BFGS-B',
                            jac=three_field_gradient_function,
                            bounds=parameters.bounds_instance,
                            args=(pt,parameters.initial_state,parameters.target_state))


    return optimised_result


def do_optimisation(max_time_array: float,
                    parameters: qsl_parameters,
                    save: bool = True,
                    return_solution: bool = False):
    assert isinstance(parameters,qsl_parameters)
    for i in tqdm(range(max_time_array.size)):

        if parameters.three_field:
            optimised_result = _do_optimisation_three_field(max_time_array[i],parameters)
        else:
            optimised_result = _do_optimisation(max_time_array[i],parameters)

        filename = parameters.name + '_time_{}'.format(max_time_array[i])
        filename_replaced = filename.replace('.','-')
        if save:
            np.save(qsl_path+filename_replaced+'.res',optimised_result.x)
        if return_solution:
            return optimised_result

def load_process_tensor(max_time: float,parameters: qsl_parameters):
    assert isinstance(parameters,qsl_parameters)

    if parameters.pt_name is None:
        if parameters.alpha is None:
            name = 'optimisation_dt{}dkmax{}esprel{}temp{}maxtime{}'.format(parameters.dt,parameters.dkmax,parameters.esprel,parameters.temperature,max_time)
        else:
            name = 'optimisation_dt{}dkmax{}esprel{}temp{}alpha{}maxtime{}'.format(parameters.dt,parameters.dkmax,parameters.esprel,parameters.temperature,parameters.alpha,max_time)

    else:
        name = parameters.pt_name + 'maxtime{}'.format(max_time)
    name_replaced = name.replace('.','-')
    pt = tempo.import_adjoint_process_tensor(qsl_path+name_replaced+'.processTensor','adjoint')
    return pt

def compute_dynamics(max_time_array:np.ndarray,parameters:qsl_parameters):
    assert isinstance(parameters,qsl_parameters)

    for i in tqdm(range(max_time_array.size)):

        cp_filename = parameters.name + '_time_{}'.format(max_time_array[i])

        cp_filename_replaced = cp_filename.replace('.','-')
        control_parameters = np.load(qsl_path+cp_filename_replaced + '.res')

        pt = load_process_tensor(max_time_array[i],parameters)
        if parameters.three_field:
            dynamics = get_three_field_dynamics(control_parameters,pt)
        else:
            dynamics = get_ARP_dynamics(control_parameters,pt,parameters.initial_state)
        dynamics_name = parameters.name + '_dynamics_maxtime_{}'.format(max_time_array[i])
        dynamics_name = dynamics_name.replace('.','-')
        dynamics.export(dynamics_path + dynamics_name + '.dynamics',overwrite=True)


def compute_initial_condition_dynamics(max_time_array:np.ndarray,parameters:qsl_parameters):
    assert isinstance(parameters,qsl_parameters)

    for i in tqdm(range(max_time_array.size)):



        pt = load_process_tensor(max_time_array[i],parameters)
        control_parameters = parameters.get_initial_guess_array(max_time_array[i])
        dynamics = get_ARP_dynamics(control_parameters,pt,parameters.initial_state)
        dynamics_name = parameters.name + 'initial_condition_dynamics_maxtime_{}'.format(max_time_array[i])
        dynamics_name = dynamics_name.replace('.','-')
        dynamics.export(dynamics_path + dynamics_name + '.dynamics',overwrite=True)


def load_dynamics_from_parameter_class(parameters: qsl_parameters,max_time):
    assert isinstance(parameters,qsl_parameters)

    cp_filename = parameters.name + '_time_{}'.format(max_time)

    cp_filename_replaced = cp_filename.replace('.','-')
    control_parameters = np.load(qsl_path+cp_filename_replaced + '.res')

    dynamics_name = parameters.name + '_dynamics_maxtime_{}'.format(max_time)

    dynamics_name = dynamics_name.replace('.','-')
    dynamics = tempo.import_dynamics(dynamics_path+dynamics_name+'.dynamics')
    return control_parameters,dynamics

def load_initial_condition_dynamics(parameters: qsl_parameters,max_time):
    assert isinstance(parameters,qsl_parameters)

    pt = load_process_tensor(max_time,parameters)

    if parameters.three_field:
        raise NotImplementedError
    elif np.array_equal(parameters.initial_state,tempo.operators.spin_dm('x-'))\
            or np.array_equal(parameters.initial_state,tempo.operators.spin_dm('x+')):
        control_parameters = control_parameters = np.zeros(2 * len(pt))
        top = np.pi / max_time
        if np.pi / max_time > parameters.bound:
            top = parameters.bound
        control_parameters[len(pt):] = top
    else:
        control_parameters = np.zeros(2 * len(pt))
        top = np.pi / max_time
        if np.pi / max_time > parameters.bound:
            top = parameters.bound
        control_parameters[:len(pt)] = top
    dynamics_name = parameters.name + 'initial_condition_dynamics_maxtime_{}'.format(max_time)

    dynamics_name = dynamics_name.replace('.','-')
    dynamics = tempo.import_dynamics(dynamics_path+dynamics_name+'.dynamics')
    return control_parameters,dynamics

def get_closed_fidelity(time_array,parameters):

    if parameters.bound_sx is not None:
        if np.array_equal(parameters.initial_state,tempo.operators.spin_dm('x+')) or np.array_equal(parameters.initial_state,tempo.operators.spin_dm('x-')):
            limiting_bound = parameters.bound
        else:
            limiting_bound = parameters.bound_sx
    else:
        limiting_bound = parameters.bound

    closed_fidelity = np.piecewise(time_array,[time_array < np.pi/limiting_bound,time_array >= np.pi/limiting_bound],[lambda t:(np.cos(limiting_bound*t / 2))**2,0])
    return closed_fidelity

def compare_results_with_QSL(max_time_array,parameters):
    assert isinstance(parameters,qsl_parameters)
    infidelities = np.zeros(max_time_array.size)
    qsl_array = np.zeros(max_time_array.size)


    for i in tqdm(range(max_time_array.size)):

        dynamics = load_dynamics_from_parameter_class(parameters,max_time_array[i])[1]
        states = dynamics.states
        final_state = states[-1,:,:]
        target_state = parameters.target_state
        infidelity = 1 - np.trace(target_state @ final_state)

        the_qsl = qsl(dynamics)
        infidelities[i] = infidelity
        qsl_array[i] = the_qsl

    fig,ax1 = plt.subplots()
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    ax1.set_xlabel('max time (ps)')
    ax1.set_ylabel('infidelity',color=colours[0])
    ax1.plot(max_time_array,infidelities,color=colours[0])
    ax1.plot(max_time_array,infidelities,'rx')

    ax1.tick_params(axis='y', labelcolor=colours[0])

    ax2 = ax1.twinx()

    ax2.set_ylabel('max time / QSL',color=colours[1])
    ax2.plot(max_time_array, qsl_array / max_time_array,color=colours[1])
    ax2.plot(max_time_array, qsl_array / max_time_array,'rx')
    ax2.tick_params(axis='y',labelcolor=colours[1])

    fig.tight_layout()
    if parameters.figname is not None:
        plt.savefig(parameters.figname + 'a.pdf')


    fig2,ax2_1 = plt.subplots()

    ax2_1.set_xlabel('max time (ps)')
    ax2_1.set_ylabel('infidelity',color=colours[0])
    ax2_1.plot(max_time_array,infidelities,color=colours[0])
    ax2_1.plot(max_time_array,infidelities,'rx')

    ax2_1.tick_params(axis='y', labelcolor=colours[0])

    ax2_2 = ax2_1.twinx()

    max_max_time = max_time_array[-1]
    min_time = max_time_array[0]
    qsl_saturation_line = np.linspace(min_time,max_max_time,100)
    ax2_2.plot(qsl_saturation_line,qsl_saturation_line,linestyle='dashdot',color='black',label='Pulse Duration')

    x_closed_fidelity = np.linspace(max_time_array[0],max_time_array[-1],100)
    closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
    ax2_1.plot(x_closed_fidelity,closed_fidelity,linestyle='dashdot',color='green',label='Closed QSL')

    ax2_2.set_ylabel('QSL (ps)',color=colours[1])
    ax2_2.plot(max_time_array, qsl_array,color=colours[1])
    ax2_2.plot(max_time_array, qsl_array,'rx')
    ax2_2.tick_params(axis='y',labelcolor=colours[1])
    ax2_1.set_title('bound = {}'.format(parameters.bound))
    plt.legend(loc=1)

    fig2.tight_layout()
    if parameters.figname is not None:
        plt.savefig(parameters.figname + 'c.pdf')


    fig3,ax3_1 = plt.subplots()
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    ax3_1.set_xlabel('max time (ps)')
    ax3_1.set_ylabel('infidelity',color=colours[0])
    ax3_1.semilogy(max_time_array,infidelities,color=colours[0])
    ax3_1.semilogy(max_time_array,infidelities,'rx')

    ax3_1.tick_params(axis='y', labelcolor=colours[0])

    ax3_2 = ax3_1.twinx()

    ax3_2.set_ylabel('max time / QSL',color=colours[1])
    ax3_2.plot(max_time_array, qsl_array / max_time_array,color=colours[1])
    ax3_2.plot(max_time_array, qsl_array / max_time_array,'rx')
    ax3_2.tick_params(axis='y',labelcolor=colours[1])

    fig3.tight_layout()
    if parameters.figname is not None:
        plt.savefig(parameters.figname + 'b.pdf')


    fig4,ax4_1 = plt.subplots()

    ax4_1.set_xlabel('max time (ps)')
    ax4_1.set_ylabel('infidelity',color=colours[0])
    ax4_1.semilogy(max_time_array,infidelities,color=colours[0])
    ax4_1.semilogy(max_time_array,infidelities,'rx')

    ax4_1.tick_params(axis='y', labelcolor=colours[0])

    ax4_2 = ax4_1.twinx()

    qsl_saturation_line = np.linspace(min_time,max_max_time,100)
    ax4_2.plot(qsl_saturation_line,qsl_saturation_line,linestyle='dashdot',color='black',label='Pulse Duration')

    x_closed_fidelity = np.linspace(max_time_array[0],max_time_array[-1],100)
    closed_fidelity = get_closed_fidelity(x_closed_fidelity,parameters)
    ax4_1.plot(x_closed_fidelity,closed_fidelity,linestyle='dashdot',color='green',label='Closed QSL')

    ax4_2.set_ylabel('QSL (ps)',color=colours[1])
    ax4_2.plot(max_time_array, qsl_array,color=colours[1])
    ax4_2.plot(max_time_array, qsl_array,'rx')
    ax4_2.tick_params(axis='y',labelcolor=colours[1])
    plt.legend(loc=1)

    fig4.tight_layout()

    if parameters.figname is not None:
        plt.savefig(parameters.figname + 'd.pdf')

    plt.show()
