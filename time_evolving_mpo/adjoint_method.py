'''
Module for computing the gradient of a problem with respect to the control parameters

'''

from datetime import time
import enum
from multiprocessing.sharedctypes import Value
from typing import Dict, Optional, Text, List


import os
import tempfile
from typing import Callable, Dict, List, Optional, Text, Tuple

import numpy as np
from numpy import complex128, float64, ndarray, number
from numpy.ma import count
from scipy.linalg import expm, norm
import tensornetwork as tn
import h5py
import time_evolving_mpo

from time_evolving_mpo.process_tensor import SimpleProcessTensor, BaseProcessTensor, compute_dynamics_from_system, _compute_dynamics
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.system import AdjointTimeDependentSystem, BaseSystem
from time_evolving_mpo import util

import time_evolving_mpo as tempo


class SimpleAdjointTensor(SimpleProcessTensor):

    def compute_derivatives_from_system_eoin(
            self,
            system: BaseSystem,
            start_time: Optional[float] = 0.0, # ask gerald why the start time has to be specified and can't be encoded in the PT
            dt: Optional[float] = None,
            initial_state: Optional[ndarray] = None,
            target_state: Optional[ndarray] = None,
            num_steps: Optional[int] = None) -> List[ndarray]:

        system_propeagator_derivatives = compute_tensors_from_system_eoin(
            process_tensor=self,
            system=system,
            start_time=start_time,
            dt=dt,
            initial_state=initial_state,
            target_state=target_state,
            num_steps=num_steps,
            record_all=True)

        self.system_propagator_derivatives = system_propeagator_derivatives
        return system_propeagator_derivatives



    def compute_derivatives_from_system(
            self,
            system: BaseSystem,
            start_time: Optional[float] = 0.0, # ask gerald why the start time has to be specified and can't be encoded in the PT
            dt: Optional[float] = None,
            initial_state: Optional[ndarray] = None,
            target_state: Optional[ndarray] = None,
            num_steps: Optional[int] = None) -> List[ndarray]:


        forward_tensors,backward_tensors =  compute_tensors_from_system(
            process_tensor=self,
            system=system,
            start_time=start_time,
            initial_state=initial_state,
            target_state=target_state,
            num_steps=num_steps)
        self.forwards=forward_tensors
        self.backwards=backward_tensors
        backlen=len(backward_tensors)
        derivlist=[]
        for i in range(len(forward_tensors)):
            forward_tensor = forward_tensors[i]
            backward_tensor = backward_tensors[backlen-1-i]
            forward_tensor[0] ^ backward_tensor[0]
            deriv = forward_tensor @ backward_tensor
            # (forward_tensors[i])[0] ^ (backward_tensors[backlen-1-i])[0]
            # deriv=forward_tensors[i] @ backward_tensors[backlen-1-i]
            derivlist.append(deriv.tensor)
        self.derivlist = derivlist # please rename
        return derivlist

    def get_derivative_times(
        self,
        start_time: Optional[float] = 0.0) -> List[ndarray]:
        '''
        Returns the times that the derivatives of the propagators are taken, for helpi  ng with finite differncing
        '''
        # this line should be in the init for an adjoint class, but i haven't figured out how to change the init
        # to inherit all the previous init
        expval_times = np.arange(0,len(self))*self.dt + start_time
        derivative_times = np.array([[t+self.dt/4.0,t+self.dt*3.0/4.0] for t in expval_times]).flatten()
        self.derivative_times = derivative_times
        return derivative_times

    # same as above, should be in init
    def get_expectation_value_times(
                        self,
                        start_time: Optional[float] = 0.0,
                        timesteps_used :Optional[int] = None) -> List[ndarray]:
        if timesteps_used is None:
            expval_times = np.arange(0,len(self))*self.dt + start_time
            expval_times_full = expval_times # ratio here is one
            # check if this can be redefined
            # expval_times_experimental = np.arange(start_time,start_time + len(self)*self.dt,self.dt)
            # difference = expval_times_experimental - expval_times
            # vector_norm = norm(difference)
            # print('vector norm = {}'.format(vector_norm))


        else:
            # don't know if this works as expected yet TODO
            # i need the full list of expval times because later i'm going to find the list of indeces
            # where the expvals are used. store the full list for later and return the list with the
            # timesteps grouped
            expval_times_full = np.arange(0,len(self))*self.dt + start_time
            expval_times = np.arange(start_time,start_time + len(self) * self.dt,timesteps_used * self.dt )
        self.expval_times = expval_times_full # stores full list
        return expval_times


    # at the moment plz give dprop_list explicitly, _derivlist can also be given explicitly, but generally will be stored in the instance
    # i can't think of a single reason why you'd want to give _derivlist seperately, since you need to create an instance to generate it in the first place
    def finite_difference_chain_rule(
        self,
        dprop_list: Optional[ndarray] = None, # the derivative of the desired parameter wrt the propagators
        pre_derivs: Optional[ndarray] = None, # deriv wrt pre node
        post_derivs: Optional[ndarray] = None, # deriv wrt post node
        time_array: Optional[ndarray] = None, # time density matrix is computed at
        timesteps_used: Optional[int] = None, # number of timesteps grouped together
        _derivlist: Optional[ndarray] = None): #derivlist can be given explicitly if known
        '''
        Computes the total derivative in the case where the derivative wrt the TEMPO system propagators is known.
        This is a very specific case where the number of derivatives == the number of half timestep propagators
        this is not the case in general
        '''


        if _derivlist is None:
            _derivlist = self.derivlist

        def combinederivs(target_derivatives,propagator_derivatives):
            assert (len(target_derivatives)==len(propagator_derivatives)), "Lists supplied have uneqal length"
            derivvalslist=[]
            for (target_deriv,propagator_deriv) in zip(target_derivatives,propagator_derivatives):
                # I need to check I've got these indices the right way
                # here's the tensornetwork version
                dtargetnode=tn.Node(target_deriv)
                dpropnode=tn.Node(propagator_deriv)
                dtargetnode[0] ^ dpropnode[0]
                dtargetnode[1] ^ dpropnode[1]
                derivvalslist.append((dtargetnode @ dpropnode).tensor+0.0)
                # if the above is right it can also be done like this
                # derivvalslist.append(np.matmul(target_deriv.T,propagator_deriv).trace())
            return derivvalslist

        def combine_derivs_single(target_deriv,propagator_deriv):
            dtargetnode=tn.Node(target_deriv)
            dpropnode=tn.Node(propagator_deriv)
            dtargetnode[0] ^ dpropnode[0]
            dtargetnode[1] ^ dpropnode[1]

            return (dtargetnode @ dpropnode).tensor

        if time_array is None:
            if dprop_list is not None:
                total_derivative_list = combinederivs(dprop_list,_derivlist)
                return total_derivative_list
            elif pre_derivs and post_derivs is not None:
                raise NotImplementedError('Ah I will do this at some stage')
            else:
                raise ValueError('Derivatives need to be given either as pre_derivs and pos_derivs explicitly orin the same list')


        # https://stackoverflow.com/questions/32191029/getting-the-indices-of-several-elements-in-a-numpy-array-at-once
        # see also link he uses
        full_time_array = self.expval_times # hmm, i haven't tested this that it actually yields what i expect
        s_full_time = np.argsort(full_time_array)
        time_indices = s_full_time[np.searchsorted(full_time_array,time_array,sorter=s_full_time)]
        # interesting behaviour, if full time array was [0,3,5] and time array was [3,2]
        # result would be [1,1] and time_array = [3,6] would give an index out of bound
        # error

        derivvalslist = []

        if dprop_list is not None:
            for counter,index in enumerate(time_indices): # returns deriv wrt pre and post nodes seperately (half timesteps)
                derivvalslist.append(combine_derivs_single(dprop_list[counter],_derivlist[index]))
            return derivvalslist
        elif pre_derivs and post_derivs is not None: # this sums the two together, to return the derivitive over one total timestep
            if timesteps_used is None:
                for counter,index in enumerate(time_indices):
                    combined_pre_deriv = combine_derivs_single(pre_derivs[counter],_derivlist[2 * index])
                    combined_post_deriv = combine_derivs_single(post_derivs[counter],_derivlist[2 * index +1])

                    whole_timestep_deriv = combined_pre_deriv + combined_post_deriv
                    derivvalslist.append(whole_timestep_deriv)
                return derivvalslist
            else:
                # check to see if the number of grouped timesteps divides evenly (last grouping has 'timesteps_used' timesteps)
                # should be full time array as working out how many timesteps to be grouped together
                modulo_result = full_time_array.size % timesteps_used
                if modulo_result == 0:
                    for counter,index in enumerate(time_indices):
                        summing_array = np.zeros(timesteps_used)

                        for i in range(timesteps_used):

                            combined_pre_deriv = combine_derivs_single(pre_derivs[counter],_derivlist[2*(index+i)])
                            combined_post_deriv = combine_derivs_single(post_derivs[counter],_derivlist[2*(index+i)+1])

                            whole_timestep_deriv = combined_pre_deriv + combined_post_deriv
                            summing_array[i] = whole_timestep_deriv
                        sum = np.sum(summing_array)
                        derivvalslist.append(sum)
                else:
                    # need to treat the last timestep seperately cause it contains a different number of timesteps
                    for counter,index in enumerate(time_indices[:-1]): # all but last timestep
                        summing_array = np.zeros(timesteps_used)

                        for i in range(timesteps_used):

                            combined_pre_deriv = combine_derivs_single(pre_derivs[counter],_derivlist[2*(index+i)])
                            combined_post_deriv = combine_derivs_single(post_derivs[counter],_derivlist[2*(index+i)+1])

                            whole_timestep_deriv = combined_pre_deriv + combined_post_deriv
                            summing_array[i] = whole_timestep_deriv
                        sum = np.sum(summing_array)
                        derivvalslist.append(sum)

                    counter = time_indices.size - 1
                    index = time_indices[counter]
                    summing_array = np.zeros(timesteps_used)
                    for i in range(modulo_result):
                        combined_pre_deriv = combine_derivs_single(pre_derivs[counter],_derivlist[2*(index+i)])
                        combined_post_deriv = combine_derivs_single(post_derivs[counter],_derivlist[2*(index+i)+1])

                        whole_timestep_deriv = combined_pre_deriv + combined_post_deriv
                        summing_array[i] = whole_timestep_deriv
                    sum = np.sum(summing_array)
                    derivvalslist.append(sum)

                return derivvalslist



        else:
            raise 'Derivatives need to be given either as pre_derivs and pos_derivs explicitly orin the same list'



def compute_tensors_from_system_eoin(
        process_tensor: BaseProcessTensor,
        system: BaseSystem,
        start_time: Optional[float] = 0.0, #not implementing
        dt: Optional[float] = None, #not implementing
        initial_state: Optional[ndarray] = None,
        target_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None,
        record_all: Optional[bool] = True) -> Dynamics: #need to delete the record all option
    """
    i'm not fully sure if this still works
    """
    # -- input parsing --
    assert isinstance(system, BaseSystem), \
        "Parameter `system` is not of type `tempo.BaseSystem`."

    hs_dim = system.dimension
    assert hs_dim == process_tensor.hilbert_space_dimension

    assert target_state is not None, 'target stte must be specified --Eoin'

    if dt is None:
        dt = process_tensor.dt
        if dt is None:
            raise ValueError("Process tensor has no timestep, "\
                + "please specify time step 'dt'.")
    try:
        __dt = float(dt)
    except Exception as e:
        raise AssertionError("Time step 'dt' must be a float.") from e

    try:
        __start_time = float(start_time)
    except Exception as e:
        raise AssertionError("Start time must be a float.") from e

    if initial_state is not None:
        assert initial_state.shape == (hs_dim, hs_dim)

    if num_steps is not None:
        try:
            __num_steps = int(num_steps)
        except Exception as e:
            raise AssertionError("Number of steps must be an integer.") from e
    else:
        __num_steps = None

    # -- compute dynamics --

    def propagators(step: int):
        """Create the system propagators (first and second half) for the
        time step `step`. """
        t = __start_time + step * __dt
        first_step = expm(system.liouvillian(t+__dt/4.0)*__dt/2.0).T
        second_step = expm(system.liouvillian(t+__dt*3.0/4.0)*__dt/2.0).T
        return first_step, second_step

    states_forward = _compute_dynamics_forwards(process_tensor=process_tensor,
                               controls=propagators,
                               initial_state=initial_state,
                               num_steps=__num_steps,
                               record_all=record_all)

    states_backwards = _compute_dynamics_backwards(process_tensor=process_tensor,
                               controls=propagators,
                               target_state=target_state,
                               num_steps=__num_steps,
                               record_all=record_all)

    assert len(states_forward) == len(states_backwards)

    number_of_derivatives = len(states_forward)

    derivative_tensors = []
    for index in range(number_of_derivatives):
        forward_node = states_forward[index]
        forward_node.add_axis_names('bond_leg','forward_state')
        backward_node = states_backwards[number_of_derivatives - index - 1]
        backward_node.add_axis_names('bond_leg','backward_state')
        # why connect these two indeces?
        # the leg dimensions match
        forward_node[0] ^ backward_node[0]
        combined_tensor = forward_node @ backward_node


        derivative_tensors.append(tn.replicate_nodes([combined_tensor])[0])

    return derivative_tensors




def compute_tensors_from_system(
        process_tensor: BaseProcessTensor,
        system: BaseSystem,
        start_time: float, #not implementing
        initial_state: Optional[ndarray] = None,
        target_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None):
    """
    TODO: rewrite this
    Compute the system dynamics for a given system Hamiltonian.

    Parameters
    ----------
    process_tensor: BaseProcessTensor
        A process tensor object.
    system: BaseSystem
        Object containing the system Hamiltonian information.

    Returns
    -------
    dynamics: Dynamics
        The system dynamics for the given system Hamiltonian
        (accounting for the interaction with the environment).
    """
    # -- input parsing --
    assert isinstance(system, BaseSystem), \
        "Parameter `system` is not of type `tempo.BaseSystem`."

    hs_dim = system.dimension
    assert hs_dim == process_tensor.hilbert_space_dimension

    assert target_state is not None, 'target stte must be specified --Eoin'


    dt = process_tensor.dt
    if dt is None:
        raise ValueError("Process tensor has no timestep, "\
            + "please specify time step 'dt'.")
    try:
        __dt = float(dt)
    except Exception as e:
        raise AssertionError("Time step 'dt' must be a float.") from e

    try:
        __start_time = float(start_time)
    except Exception as e:
        raise AssertionError("Start time must be a float.") from e

    if initial_state is not None:
        assert initial_state.shape == (hs_dim, hs_dim)

    if num_steps is not None:
        try:
            __num_steps = int(num_steps)
        except Exception as e:
            raise AssertionError("Number of steps must be an integer.") from e
    else:
        __num_steps = None

    # -- compute dynamics --

    def propagators(step: int):
        """Create the system propagators (first and second half) for the
        time step `step`. """
        t = __start_time + step * __dt
        first_step = expm(system.liouvillian(t+__dt/4.0)*__dt/2.0).T
        second_step = expm(system.liouvillian(t+__dt*3.0/4.0)*__dt/2.0).T
        return first_step, second_step

    states_forward = _compute_dynamics_forwards(process_tensor=process_tensor,
                               controls=propagators,
                               initial_state=initial_state,
                               num_steps=__num_steps)

    states_backwards = _compute_dynamics_backwards(process_tensor=process_tensor,
                               controls=propagators,
                               target_state=target_state,
                               num_steps=__num_steps)

    return states_forward,states_backwards #derivative_tensors



def _compute_dynamics_forwards(
        process_tensor: BaseProcessTensor,
        controls: Callable[[int], Tuple[ndarray, ndarray]],
        initial_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None,
        record_all: Optional[bool] = True) -> List[ndarray]:
    """The same contraction algorithm to solve for the density matrix as a function of time,
    however now saves the PT at the times that will yield the derivatives, and no longer computes the states """
    hs_dim = process_tensor.hilbert_space_dimension

    initial_tensor = process_tensor.get_initial_tensor()
    assert (initial_state is None) ^ (initial_tensor is None), \
        "Initial state must be either (exclusively) encoded in the " \
        + "process tensor or given as an argument."
    if initial_tensor is None:
        initial_tensor = util.add_singleton(
            initial_state.reshape(hs_dim**2), 0)

    current = tn.Node(initial_tensor)
    current_bond_leg = current[0]
    current_state_leg = current[1]

    currents_forward_list = []

    if num_steps is None: #again why do i want this to be optional
        __num_steps = len(process_tensor)
    else:
        __num_steps = num_steps

    for step in range(__num_steps):
        # this is where the compute expectation values *was*
        try:
            mpo = process_tensor.get_mpo_tensor(step)
        except Exception as e:
            raise ValueError("The process tensor is not long enough") from e
        if mpo is None:
            raise ValueError("Process tensor has no mpo tensor "\
                +f"for step {step}.")
        mpo_node = tn.Node(mpo)
        pre, post = controls(step)
        pre_node = tn.Node(pre)
        post_node = tn.Node(post)
        lam = process_tensor.get_lam_tensor(step)
        # first state saved is just the initial product state between the system and the bath
        # states saved at the start of the loop so that the last post is never included in the forward prop
        if lam is None:
            currents_forward_list.append(tn.replicate_nodes([current])[0]) # save the state just before the **step** pre tensor
            current_bond_leg ^ mpo_node[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ mpo_node[2]

            current_bond_leg = mpo_node[1]
            current_state_leg = mpo_node[3]

            # now calculate the ADT after one half step
            current = current @ pre_node @ mpo_node
            current.add_axis_names(['0','state_forward']) # for debugging

            currents_forward_list.append(tn.replicate_nodes([current])[0])# save the state just before the **step** post tensor
            current_state_leg ^ post_node[0]
            current_state_leg = post_node[1]

            current = current @ post_node
            current.add_axis_names(['0','state_forward'])



        else:
            raise NotImplementedError('I havent sorted the lambdas yet --Eoin')

            lam_node = tn.Node(lam)
            current_bond_leg ^ mpo_node[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ mpo_node[2]
            mpo_node[1] ^ lam_node[0]
            mpo_node[3] ^ post_node[0]
            current_bond_leg = lam_node[1]
            current_state_leg = post_node[1]
            current = current @ pre_node @ mpo_node @ lam_node @ post_node
            currents_forward_list = currents_forward_list.append(current)


    return currents_forward_list

def _compute_dynamics_backwards(
        process_tensor: SimpleAdjointTensor,
        controls: Callable[[int], Tuple[ndarray, ndarray]],
        target_state: ndarray,
        num_steps: Optional[int] = None, # hmmm, should i be allowed to do that?
        record_all: Optional[bool] = True) -> List[ndarray]:
    """Same as the forward prop but now the algorithm starts with the target
    state and propagates backwards"""
    hs_dim = process_tensor.hilbert_space_dimension


    transposed_ts = target_state.T # transpose is equivalent to conjugate (see pdf for reason why)
    target_state_tensor = transposed_ts.reshape(hs_dim**2)



    if num_steps is None:
        __num_steps = len(process_tensor)
    else:
        __num_steps = num_steps


    final_cap=process_tensor.get_cap_tensor(__num_steps)


    current=tn.Node(np.outer(final_cap,target_state_tensor))

    current_bond_leg = current[0]
    current_state_leg = current[1]
    currents_backwards_list = []


    for reversed_step in reversed(range(__num_steps)):


        try:
            mpo = process_tensor.get_mpo_tensor(reversed_step)
        except Exception as e:
            raise ValueError("The process tensor is not long enough") from e
        if mpo is None:
            raise ValueError("Process tensor has no mpo tensor "\
                +f"for reversed_step {reversed_step}.")
        mpo_node = tn.Node(mpo,name='mpo')
        pre, post = controls(reversed_step)
        pre_node = tn.Node(pre,name='pre')
        post_node = tn.Node(post,name='post')

        lam = process_tensor.get_lam_tensor(reversed_step)
        # internal structure is the same configuration as the forward propagation
        # only the output legs are reversed
        if lam is None:
            # saving times at the start of the loop
            currents_backwards_list.append(tn.replicate_nodes([current])[0]) #post node missing
            post_node[0] ^ mpo_node[3]
            current_bond_leg ^ mpo_node[1]
            current_state_leg ^ post_node[1] # now connects to post_node rather than pre_node
            # these two are unchanged


            current_bond_leg = mpo_node[0]
            current_state_leg = mpo_node[2]
            current = current @ post_node @ mpo_node
            current.add_axis_names(['0','state_backwards'])
            currents_backwards_list.append(tn.replicate_nodes([current])[0]) # pre node missing

            current_state_leg ^ pre_node[1] # now connects to the pre node rather than post node
            current_state_leg = pre_node[0]
            # current_bond_leg = mpo_node[0]

            current = current @ pre_node
            current.add_axis_names(['0','state_backwards'])

            # now the output leg is a pre_node as opposed to a post_node

            # the very first pre node is never saved, as the last backprop is with the 'first' pre node missing

        else:
            raise NotImplementedError('you done goofed. Regards --Eoin')
            lam_node = tn.Node(lam)
            current_bond_leg ^ mpo_node[0]
            current_state_leg ^ pre_node[0]
            pre_node[1] ^ mpo_node[2]
            mpo_node[1] ^ lam_node[0]
            mpo_node[3] ^ post_node[0]
            current_bond_leg = lam_node[1]
            current_state_leg = post_node[1]
            current = current @ pre_node @ mpo_node @ lam_node @ post_node


    return currents_backwards_list

