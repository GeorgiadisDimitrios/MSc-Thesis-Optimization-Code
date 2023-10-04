import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/eobutler/dev/adjoint_TimeEvolvingMPO_pre/scripts")
import time_evolving_mpo as tempo
from scipy.linalg import expm

from time import time

from scripts.cost_function import get_ARP_hamiltonian

from typing import Union

def pure_dephasing(hx :Union[np.ndarray,int,float],
                hz :Union[np.ndarray,int,float],
                dt :float,
                dissipator_rate :Union[np.ndarray,int,float],
                max_time :float,
                get_dynamics :bool= True
            ) -> np.ndarray:
    '''
    generates liouvillian superoperators for pure dephasing
    '''


    dt = 0.05
    max_time = 5.0

    length = int(max_time / dt)

    assert hx.size == length,\
        'one hx for every timestep'
    assert hz.size == length,\
        'one hz for every timestep'

    initial_state = tempo.operators.spin_dm('x-')
    initial_state_vec = initial_state.reshape(initial_state.shape[0]**2)

    if isinstance(dissipator_rate,float) or isinstance(dissipator_rate,int):

        def get_tensor(dissipator_rate,step) -> np.ndarray:
            tensor = np.identity(4)
            exp_value = np.exp(-dissipator_rate * dt)
            tensor[1,1] = exp_value
            tensor[2,2] = exp_value

            return tensor

    else:
        assert isinstance(dissipator_rate,np.ndarray), \
            'dissipator rate must be given as type ndarray or int or float'
        assert length == dissipator_rate.size,\
            ('if the dissipator rate is given as an ndarray, one rate for each '
            'timestep must be provided')
        def get_tensor(dissipator_rate,step) -> np.ndarray:
            tensor = np.identity(4)
            exp_value = np.exp(-dissipator_rate * dt)
            tensor[1,1] = exp_value[step]
            tensor[2,2] = exp_value[step]

            return tensor

    states = []

    def get_pre_post_nodes(h_x,h_z):
        ham = h_x * 0.5 * tempo.operators.sigma('x') + h_z * 0.5 * tempo.operators.sigma('z')
        liouvillian = -1j * tempo.util.commutator(ham)
        # the same cause hamiltonina is piecewise continuous across a single timestep
        first_step = expm(liouvillian*dt/2.0).T
        second_step = expm(liouvillian*dt/2.0).T

        return first_step,second_step

    current = initial_state_vec
    for i in range(length):
        first,second = get_pre_post_nodes(hx[i],hz[i])
        current = np.dot(current,first)
        mpo = get_tensor(dissipator_rate,i)
        current = np.dot(current,mpo)
        current = np.dot(current,second)

        if get_dynamics:
            states.append(np.copy(
                current.reshape(initial_state.shape[0],initial_state.shape[0])))
    if get_dynamics is False:
        states.append(np.copy(
                current.reshape(initial_state.shape[0],initial_state.shape[0])))

    return np.array(states)

def pure_dephasing_pt_mpo(dt,
                dissipator_rate,
                max_time,
            ) -> tempo.SimpleAdjointTensor:
    '''
    generates liouvillian superoperators for pure dephasing
    '''

    tensor = np.identity(4)
    exp_value = np.exp(-dissipator_rate * dt)
    tensor[1,1] = exp_value
    tensor[2,2] = exp_value
    tensor = tensor[np.newaxis,:,:]
    tensor = tensor[np.newaxis,:,:,:]
    # tensor = tensor_2
    # tensor.reshape((1,1,4,4))
    print(tensor)
    pt = tempo.SimpleAdjointTensor(
                    hilbert_space_dimension=2,
                    dt=dt)

    times = np.arange(0,max_time,dt)

    for i in range(times.size):
        pt.set_mpo_tensor(i,tensor)
        pt.set_cap_tensor(i,np.array([1]))
    pt.set_cap_tensor(times.size,np.array([1]))

    return pt

def dynamics_test_2(gamma):

    t = np.arange(0.05,5.05,0.05)

    hz = np.ones(t.size) * np.pi / 5
    hx = np.zeros(t.size)
    time1 = time()
    states = pure_dephasing(
        hx=hx,
        hz=hz,
        dt=0.05,
        dissipator_rate=gamma,
        max_time=5.0)
    time2 = time()
    print(time2-time1)

    sigma_x = np.zeros(len(states),dtype=np.complex128)
    sigma_y = np.zeros(len(states),dtype=np.complex128)
    sigma_z = np.zeros(len(states),dtype=np.complex128)

    for i in range(len(states)):
        sigma_x[i] = np.trace(np.matmul(tempo.operators.sigma('x'),states[i]))
        sigma_y[i] = np.trace(np.matmul(tempo.operators.sigma('y'),states[i]))
        sigma_z[i] = np.trace(np.matmul(tempo.operators.sigma('z'),states[i]))

    plt.plot(t,sigma_x)
    plt.plot(t,sigma_y)
    plt.plot(t,sigma_z)

    plt.show()




def dynamics_test(gamma):


    dt = 0.05
    max_time = 5.0

    length = int(max_time / dt)

    initial_state = tempo.operators.spin_dm('x-')

    hz = np.ones(length) * np.pi / 5
    hx = np.zeros(length)
    control_parameters = np.concatenate([hx,hz])

    pt = pure_dephasing_pt_mpo(dt,gamma,max_time)

    hamiltonian = get_ARP_hamiltonian(control_parameters,pt)
    system = tempo.AdjointTimeDependentSystem(hamiltonian)

    dynamics = pt.compute_dynamics_from_system(system,initial_state=initial_state)

    t,sigma_x = dynamics.expectations(tempo.operators.sigma('x'))
    t,sigma_y = dynamics.expectations(tempo.operators.sigma('y'))
    t,sigma_z = dynamics.expectations(tempo.operators.sigma('z'))

    plt.plot(t,sigma_x)
    plt.plot(t,sigma_y)
    plt.plot(t,sigma_z)

# dynamics_test(1)
# plt.show()


# test = np.linspace(0,1,100)

# ibm_gammas = np.load('gammas.npy')
# dynamics_test_2(0.175)


