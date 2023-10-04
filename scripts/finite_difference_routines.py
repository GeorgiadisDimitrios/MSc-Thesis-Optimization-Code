'''
A place to store all the various finite difference routines

'''

import numpy as np

from time_evolving_mpo.adjoint_method import SimpleAdjointTensor
from time_evolving_mpo.system import TimeDependentSystem
import time_evolving_mpo as tempo
# from tqdm.contrib import tenumerate
from numpy import complex128, float64, ndarray, matmul, zeros, sum, array
from scipy.linalg import expm

from jax.scipy.linalg import expm as jexpm
# from jax_routines import commutator
from jax import grad, jacfwd
from jax import numpy as jnp

from typing import Callable, Optional

import matplotlib.pyplot as plt
def time_dependent_finite_difference(
                    process_tensor: tempo.SimpleAdjointTensor,
                    system:tempo.AdjointTimeDependentSystem,
                    times :ndarray, 
                    operator :ndarray,
                    h :float64,
                    timesteps_used: Optional[int] = None):

    ham=system.hamiltonian

    def dpropagator(system,
        t,
        dt,
        op,
        h):
        temp=h
        h=temp
        '''
        deriv of propagator wrt the pre node and the post node
        '''

        post_liouvillian_plus_h=-1j * tempo.util.commutator(ham(t+dt*3.0/4.0)+h*op)
        post_liouvillian_minus_h=-1j * tempo.util.commutator(ham(t+dt*3.0/4.0)-h*op)

        post_propagator_plus_h=expm(post_liouvillian_plus_h*dt/2.0).T
        post_propagator_minus_h=expm(post_liouvillian_minus_h*dt/2.0).T

        postderiv=(post_propagator_plus_h-post_propagator_minus_h)/(2.*h)

        pre_liouvillian_plus_h=-1j * tempo.util.commutator(ham(t+dt*1.0/4.0)+h*op)
        pre_liouvillian_minus_h=-1j * tempo.util.commutator(ham(t+dt*1.0/4.0)-h*op)

        pre_propagator_plus_h=expm(pre_liouvillian_plus_h*dt/2.0).T
        pre_propagator_minus_h=expm(pre_liouvillian_minus_h*dt/2.0).T
        prederiv=(pre_propagator_plus_h-pre_propagator_minus_h)/(2.*h)
        return prederiv,postderiv

    pre_derivs = []
    post_derivs = []
    for step in range(times.size):
        prederiv,postderiv=dpropagator(system,times[step],process_tensor.dt,operator,h)
        pre_derivs.append(prederiv)
        post_derivs.append(postderiv)

    final_derivs = process_tensor.finite_difference_chain_rule(time_array=times,pre_derivs=pre_derivs,post_derivs=post_derivs,timesteps_used=timesteps_used)
    final_derivs_array = -1 * array(final_derivs) # infidelity
    return final_derivs_array


def jax_auto_grad(
                    process_tensor: tempo.SimpleAdjointTensor,
                    system:tempo.AdjointTimeDependentSystem,
                    times :ndarray, 
                    operator :ndarray,
                    h :float64,
                    timesteps_used: Optional[int] = None):


    ham=system.hamiltonian


    def dpropagator(system,
        t, # expectation value times
        dt,
        op,
        h):
        '''
        deriv of propagator wrt the pre node and the post node
        '''



        post_liouvillian_plus_h=-1j * tempo.util.commutator(ham(t+dt*3.0/4.0)+h*op)
        post_liouvillian_minus_h=-1j * tempo.util.commutator(ham(t+dt*3.0/4.0)-h*op)

        post_propagator_plus_h=expm(post_liouvillian_plus_h*dt/2.0).T
        post_propagator_minus_h=expm(post_liouvillian_minus_h*dt/2.0).T

        postderiv=(post_propagator_plus_h-post_propagator_minus_h)/(2.*h)

        pre_liouvillian_plus_h=-1j * tempo.util.commutator(ham(t+dt*1.0/4.0)+h*op)
        pre_liouvillian_minus_h=-1j * tempo.util.commutator(ham(t+dt*1.0/4.0)-h*op)

        pre_propagator_plus_h=expm(pre_liouvillian_plus_h*dt/2.0).T
        pre_propagator_minus_h=expm(pre_liouvillian_minus_h*dt/2.0).T
        prederiv=(pre_propagator_plus_h-pre_propagator_minus_h)/(2.*h)
        return prederiv,postderiv

    import sys
    sys.exit()

def pulse_area_finite_difference(
                process_tensor: tempo.SimpleAdjointTensor,
                hamiltonian: Callable[[float,float], ndarray],
                pulse_area: ndarray,
                h):
    dt = process_tensor.dt
    def d_propagator_d_area(t,pulse_area,h):
        post_liouvillian_plus_h = -1j * tempo.util.commutator(hamiltonian(t,area=pulse_area+h))
        post_liouvillian_minus_h = -1j * tempo.util.commutator(hamiltonian(t,area=pulse_area-h))


        post_propagator_plus_h = expm(post_liouvillian_plus_h*dt/2.0).T
        post_propagator_minus_h = expm(post_liouvillian_minus_h*dt/2.0).T

        postderiv = (post_propagator_plus_h-post_propagator_minus_h)/(2.*h)
        return postderiv


    pt_times = process_tensor.get_derivative_times()
    final_derivative_array = zeros(pulse_area.size)
    for i in range(pulse_area.size):
        dprop_list = []
        for j in range(pt_times.size):
            dprop_list.append(d_propagator_d_area(pt_times[j],pulse_area[i],h))
        result = process_tensor.finite_difference_chain_rule(dprop_list,time_array=pt_times)
        final_derivative_array[i] = sum(result)

    return final_derivative_array

