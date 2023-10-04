from path import data_path as path
from path import qsl_path
import numpy as np

optimisation_result_dict = {
    '1Kv1':{'control_parameters':path+'control_parameters_good.res','process_tensor':path+'details_pt_tempo_dt0-07dkmax50esprel1e-07.processTensor'},
    '20Kv1':{'control_parameters':path+'control_parameters_20K_0-5.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp20.processTensor'},
    '5Kv1':{'control_parameters':path+'control_parameters_5Kv1.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp5.processTensor'},
    # all of the above failed to converged
    # this converged
    '5Ksuccess':{'control_parameters':path+'control_parameters_5K_success.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp5.processTensor'}, # note change, TODO examine if necessary
    # same optimisation result, just a higher precision PT
    '5Ksuccess_high_precision':{'control_parameters':path+'control_parameters_5K_success.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-09temp5.processTensor'},
    # converged
    '5Kv2':{'control_parameters':path+'control_parameters_5Kv2.res','process_tensor':path+'details_pt_tempo_dt0-04dkmax80esprel1e-07temp5.processTensor'},
    '20K_success':{'control_parameters':path+'control_parameters_20K_success.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp20.processTensor'},
    '20K_success_higher_precision':{'control_parameters':path+'control_parameters_20K_success_higher_precision.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'},
    '20K_determanistic_1':{'control_parameters':path+'determanistic_calculation_1.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'},
    '20K_determanistic_2':{'control_parameters':path+'determanistic_calculation_2.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'},
    # before the floor division bug
    # after
    '20K_V3':{'control_parameters':path+'20K_V3.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'},
    '20K_V3_determanistic_test':{'control_parameters':path+'20K_V3_determanistic_test.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'},
    '20K_bound':{'control_parameters':path+'20K_bound.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'}, # bound between -5 and +5
    '20K_bound_2':{'control_parameters':path+'20K_bound_2.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'}, # bound between -pi/5 and +pi/5, didn't converge, stopped after 8 iterations, optimiser achieved no improvement in infidelity
    '1K_V2':{'control_parameters':path+'1K_V2.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp1.processTensor'}, #unbound, number of required iterations low ~180
    '5k_3ps':{'control_parameters':path+'5k_3ps.res','process_tensor':path+'details_pt_tempo_dt0-04dkmax80esprel1e-07temp5_taumax_3.processTensor'},
    'initial_guess_1k':{'control_parameters':path+'initial_guess_1.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp1.processTensor'},
    'initial_guess_5k':{'control_parameters':path+'initial_guess_5.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp5.processTensor'},
    'initial_guess_20k':{'control_parameters':path+'initial_guess_20.res','process_tensor':path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp20.processTensor'},
    'initial_guess_no_coupling':{'control_parameters':path+'initial_guess_1.res','process_tensor':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07no_coupling.processTensor'},
    'QSL_optimisation_0-5ps':{'control_parameters':qsl_path+'optimised_result_0-5.res','process_tensor':qsl_path+'optimisation_dt0-05dkmax60esprel1e-07temp5maxtime0-5.processTensor'},
    'QSL_global_qsl_bound_2':{'control_parameters':path+'QSL_global_qsl_bound_2.res','process_tensor':qsl_path+'optimisation_dt0-05dkmax60esprel1e-07temp5maxtime1-57.processTensor'},


    }

precision_analysis_dictionary = {
    '5Kexample':{'base_pt':path+'details_pt_tempo_dt0-05dkmax60esprel1e-07temp5.processTensor',
            'control_parameters':path+'control_parameters_5K_success.res',
            'comparison_PTs':[path+'details_pt_tempo_dt0-05dkmax60esprel1e-08temp5.processTensor',
                    path+'details_pt_tempo_dt0-05dkmax120esprel1e-07temp5.processTensor',
                    path+'details_pt_tempo_dt0-025dkmax100esprel1e-07temp5.processTensor',
                    path+'details_pt_tempo_dt0-025dkmax120esprel1e-07temp5.processTensor',
                    path+'details_pt_tempo_dt0-025dkmax120esprel1e-08temp5.processTensor'
                    ]},
    'bound_1_qsl':{'base_pt':qsl_path+'optimisation_dt0-05dkmax60esprel1e-07temp5maxtime3-6.processTensor',
            'control_parameters':qsl_path+'bound_1_time_3-6.res',
            'comparison_PTs':[qsl_path+'optimisation_dt0-025dkmax100esprel1e-07temp5maxtime3-6.processTensor',
                    qsl_path+'optimisation_dt0-05dkmax100esprel1e-07temp5maxtime3-6.processTensor',
                    qsl_path+'optimisation_dt0-05dkmax60esprel1e-09temp5maxtime3-6.processTensor',
                    qsl_path+'optimisation_dt0-025dkmax100esprel1e-08temp5maxtime3-6.processTensor']

    }

    }


quantum_speed_limit_dict = {
    'bound_1':{'bound':1,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field'},
    'bound_2':{'bound':2,'values':np.array([1.0,1.25,1.3,1.4,1.5,1.6,1.7,1.75,1.8,1.9,2.0,2.1,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5]),'type':'two_field'},
    'bound_5':{'bound':5,'values':np.array([0.25,0.5,0.6,0.7,0.75,0.8,0.9,1,1.1,1.2,1.25,1.3,1.4,1.5,2,3,5]),'type':'two_field'},
    'bound_0_5':{'bound':0.5,'values':np.array([4,4.5,5,5.5,5.6,5.8,6,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7,7.1,7.2,7.5,8,9]),'type':'two_field'},
    'bound_1_three_field':{'bound':1,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'three_field'},
    'bound_2_three_field':{'bound':2,'values':np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5]),'type':'three_field'},
    'bound_1_sx_initial_state':{'bound':1,'values':np.array([2,2.2,2.4,2.6,2.8,3.0,3,3.2,3.4,3.5,4.0,4.5,5]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_5_sx_initial_state':{'bound':5,'values':np.array([0.5,0.6,0.7,0.75,0.8,0.9,1,1.1,1.2,2,3]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    # this next one worked, but i accidently had a bug where the initial state and the target state were the same
    'bound_1_sz_5_sx':{'bound':1,'bound_sx':5,'values':np.array([2,2.2,2.4,2.6,2.8,3.0,3,3.2,3.4,3.5,4.0,4.5,5]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_1_sz_5_sx_v2':{'bound':1,'bound_sx':5,'values':np.array([2,2.2,2.4,2.6,2.8,3.0,3,3.2,3.4,3.5,4.0,4.5,5]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_1_sz_5_sx_strong_alpha':{'bound':1,'bound_sx':5,'values':np.array([2,2.2,2.4,2.6,2.8,3.0,3,3.2,3.4,3.5,4.0,4.5,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'test'},
    'bound_1_higher_precision':{'bound':1,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','pt_parameters':{'dt':0.05,'dkmax':60,'esprel':1e-08,'temp':5}},
    # 'bound_0_5_sz_5_sx_5':{'bound':0.5,'bound_sx':5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_0_5_sz_5_sx_5':{'bound':0.5,'bound_sx':5,'values':np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.2,5.4,5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    # 'bound_0_5_sz_5_sx_5':{'bound':0.5,'bound_sx':5,'values':np.array([1,1.5,2,2.5]),'type':'two_field','initial_state':'x-','target_state':'x+'},

    'bound_0_5_sz_5_sx_5alphax5':{'bound':0.5,'bound_sx':5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_parameters':{'dt':0.05,'dkmax':60,'esprel':1e-08,'temp':5,'alpha':0.63}},# 5*0.126 which is old alpha
    'bound_0_5_sz_5_sx_5alphax2':{'bound':0.5,'bound_sx':5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_parameters':{'dt':0.05,'dkmax':60,'esprel':1e-08,'temp':5,'alpha':0.252}},# 2*0.126 which is old alpha
    'bound_0_5_sz_5_sx_5alphax2_intermediate':{'bound':0.5,'bound_sx':5,'values':np.array([5.8,6.2]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_parameters':{'dt':0.05,'dkmax':60,'esprel':1e-08,'temp':5,'alpha':0.252}},# 2*0.126 which is old alpha
    'bound_0_5_sz_5_sx_better_ic':{'bound':0.5,'bound_sx':5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_5_sz_0_5_sx':{'bound':5,'bound_sx':0.5,'values':np.array([3,3.5,4,4.5,5,5.2,5.4,5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field'},
    # 'bound_5_sz_0_5_sx':{'bound':5,'bound_sx':0.5,'values':np.array([3,3.5,4,4.5,5,5.2,5.4,5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8]),'type':'two_field'},

    # 'bound_5_sz_0_5_sx':{'bound':5,'bound_sx':0.5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8]),'type':'two_field'},

    'bound_5_sz_0_5_sx_extra':{'bound':5,'bound_sx':0.5,'values':np.array([3,3.5,4,4.5,5,5.2,5.4]),'type':'two_field'},
    'bound_0_5_sz_5_sx_better_ic_rerun':{'bound':0.5,'bound_sx':5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_0_5_sz_5_sx_5_rerun':{'bound':0.5,'bound_sx':5,'values':np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.2,5.4,5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+'},
    'bound_0_5_sz_5_sx_ic_sx':{'bound':0.5,'bound_sx':5,'values':np.array([5.6,5.8,6.1,6.2,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.5,9]),'type':'two_field','initial_state':'x-','target_state':'x+','initial_condition':'pi_hx'},
    'bound_1_sz_5_sx_V3':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+',},
    'non_mark_test':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5]),'type':'two_field','initial_state':'x-','target_state':'x+',},
    'bound_1_sz_5_sx_V4':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+',},
    'pure_dephasing_v1':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0.85','dissipator_rate':0.85},
    'pure_dephasing_v2':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0.15','dissipator_rate':0.15},
    'pure_dephasing_v3':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0.075','dissipator_rate':0.075},
    'pure_dephasing_v4':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0.015','dissipator_rate':0.015},
    'pure_dephasing_ic_guess':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0.015','dissipator_rate':0.015,'initial_condition':np.array([0.1,0.1])},
    'pure_dephasing_ic_guess_2':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0.015','dissipator_rate':0.015,'initial_condition':'markov_random'},
    'closed_system_optimisation':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0-0','dissipator_rate':0.0,'initial_condition':'markov_random'},
    'closed_system_optimisation_better_IC':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','pt_name':'pure_dephasing_dissipator_rate_0-0','dissipator_rate':0.0},
    'bound_1_sz_5_sx_nonlinear_IC':{'bound':1,'bound_sx':5,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'x-','target_state':'x+','initial_condition':'markov_random'},
    'bound_5_sz_1_sx':{'bound':5,'bound_sx':1,'values':np.array([1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3,3.1,3.2,3.25,3.3,3.4,3.5,3.6,3.7,3.75,3.8,3.9,4,4.25,4.5,4.75,5]),'type':'two_field','initial_state':'z-','target_state':'z+',},









}

