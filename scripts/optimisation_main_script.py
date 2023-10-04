from data_dictionary import quantum_speed_limit_dict
import time_evolving_mpo as tempo

from quantum_speed_limit_optimisation import qsl_optimisation, get_qsl_parameter_class_from_dict

from slideshow_qsl_generator import do_everything


generate_process_tensors = False
do_optimisation = False
compute_dynamics = True
plot_graphs = True

make_slideshow = True

compute_initial_condition_dynamics = True


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# QSL string
qsl_string = 'bound_5_sz_1_sx'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


parameters_instance = get_qsl_parameter_class_from_dict(qsl_string)

qsl_dictionary = quantum_speed_limit_dict[qsl_string]
loop_array = qsl_dictionary['values']


if make_slideshow:
    do_everything(loop_array,parameters_instance)



else:
    qsl_optimisation(parameters_instance,loop_array,
                                    plot_graphs=plot_graphs,
                                    run_optimisation=do_optimisation,
                                    compute_dyn=compute_dynamics,
                                    generate_PTs=generate_process_tensors,
                                    ic_compute_dyn = compute_initial_condition_dynamics
                                    )


