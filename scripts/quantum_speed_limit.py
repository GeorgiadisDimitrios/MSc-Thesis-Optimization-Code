import numpy as np

import time_evolving_mpo as tempo

from scipy.linalg import norm
from scipy.integrate import trapezoid


from path import qsl_path,dynamics_path



def qsl(dynamics):
    assert isinstance(dynamics,tempo.Dynamics), 'dynamics must be of tempo dynamics'

    states=dynamics.states
    times = dynamics.times

    difference = states[1:,:,:] - states[:-1,:,:]
    dt_differences = times[1:] - times[:-1]
    dt = np.mean(dt_differences)
    derivatives = difference / dt
    norms = np.zeros(derivatives.shape[0])

    for i in range(norms.size):
        norms[i] = norm(derivatives[i,:,:],ord=2)

    integral = trapezoid(norms,x=times[:-1])
    Lambda_op = integral / (times[-1]-times[0])

    overlap = np.trace(states[0] @ states[-1])
    bures_angle = np.arccos(np.sqrt(overlap))
    sin2_bures_angle = (np.sin(bures_angle))**2

    QSL = sin2_bures_angle / Lambda_op
    return QSL

def target_state_qsl(dynamics,target_state):

    assert isinstance(dynamics,tempo.Dynamics), 'dynamics must be of tempo dynamics'

    states=dynamics.states
    times = dynamics.times
    difference = states[1:,:,:] - states[:-1,:,:]
    dt_differences = times[1:] - times[:-1]
    dt = np.mean(dt_differences)
    derivatives = difference / dt
    norms = np.zeros(derivatives.shape[0])

    for i in range(norms.size):
        norms[i] = norm(derivatives[i,:,:],ord=2)

    integral = trapezoid(norms,x=times[:-1])
    Lambda_op = integral / (times[-1]-times[0])

    overlap = np.trace(states[0] @ target_state)
    bures_angle = np.arccos(np.sqrt(overlap))
    sin2_bures_angle = (np.sin(bures_angle))**2

    QSL = sin2_bures_angle / Lambda_op
    return QSL

def get_dynamics_from_parameters(bound,max_time):



    cp_filename = 'optimised_result_bound{}time_{}'.format(bound, max_time)

    cp_filename_replaced = cp_filename.replace('.','-')
    control_parameters = np.load(qsl_path+cp_filename_replaced + '.res')



    dynamics_name = 'qsl_dynamics_bound_{}_maxtime_{}'.format(bound,max_time)
    dynamics_name = dynamics_name.replace('.','-')
    dynamics = tempo.import_dynamics(dynamics_path+dynamics_name+'.dynamics')
    return control_parameters,dynamics