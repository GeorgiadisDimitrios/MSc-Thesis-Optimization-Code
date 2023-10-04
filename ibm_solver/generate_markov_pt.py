
import sys
sys.path.append("/home/eobutler/dev/adjoint_TimeEvolvingMPO_pre/scripts")

import numpy as np

from scripts.generate_adjoint_pt import generate_pure_dephasing_pt

from scripts.quantum_speed_limit_optimisation import qsl_parameters, get_qsl_parameter_class_from_dict
from scripts.data_dictionary import quantum_speed_limit_dict

name = 'closed_system_optimisation_better_IC'

qsl_dict = quantum_speed_limit_dict[name]

parameters = get_qsl_parameter_class_from_dict(name)

max_times = qsl_dict['values']

for i in range(max_times.size):
    generate_pure_dephasing_pt(parameters.dt,
        qsl_dict['dissipator_rate'],
        max_times[i])



