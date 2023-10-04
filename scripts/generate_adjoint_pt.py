
import time_evolving_mpo as tempo

import numpy as np
import matplotlib.pyplot as plt

from path import data_path, qsl_path
from warnings import warn


def sd(w):
    return ((0.002) * w)/(1 + (w**2/(800*np.pi))**2)


def generate_pt_QD(dt,
                dkmax,
                esprel,
                _temperature, # units of Kelvin
                plot_correlations=False,
                disable_coupling = False,
                alpha = 0.126):
    # generate the process tensor and save to file
    omega_cutoff = 3.04
    if disable_coupling is True:
        alpha = 0.0

    temperature = _temperature * 0.1309 # 1K = 0.1309/ps

    correlations = tempo.PowerLawSD(alpha=alpha,
                                    zeta=3,
                                    cutoff=omega_cutoff,
                                    cutoff_type='gaussian',
                                    max_correlation_time=5.0,
                                    temperature=temperature)
    bath = tempo.Bath(tempo.operators.sigma("z")/2.0, correlations)

    pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=esprel)

    if plot_correlations:
        tempo.helpers.plot_correlations_with_parameters(correlations, pt_tempo_parameters)
        figname = 'correlations_dt{}dkmax{}esprel{}temp{}'.format(dt,dkmax,esprel,_temperature)
        figname_replaced = figname.replace('.','-')
        plt.savefig(figname_replaced+'.pdf')
        plt.show()

    process_tensor = tempo.pt_tempo_compute(bath=bath,
                                            start_time=0,
                                            end_time=5,
                                            parameters=pt_tempo_parameters)

    name = 'details_pt_tempo_dt{}dkmax{}esprel{}temp{}'.format(dt,dkmax,esprel,_temperature)
    if alpha != 0.126:
        name = 'details_pt_tempo_dt{}dkmax{}esprel{}alpha{}temp{}'.format(dt,dkmax,esprel,alpha,_temperature)

    name_replaced = name.replace('.','-')
    process_tensor.export(data_path+name_replaced+".processTensor",overwrite=True)

def generate_pt_QSL_QD(dt,
                dkmax,
                esprel,
                _temperature, # units of Kelvin
                max_time,
                alpha = 0.126,
                name = None,
                plot_correlations=False,
                disable_coupling = False):
    # generate the process tensor and save to file
    omega_cutoff = 3.04
    if disable_coupling is True:
        alpha = 0.0

    temperature = _temperature * 0.1309 # 1K = 0.1309/ps

    correlations = tempo.PowerLawSD(alpha=alpha,
                                    zeta=3,
                                    cutoff=omega_cutoff,
                                    cutoff_type='gaussian',
                                    max_correlation_time=5.0,
                                    temperature=temperature)
    bath = tempo.Bath(tempo.operators.sigma("z")/2.0, correlations)

    pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=esprel)

    if plot_correlations:
        tempo.helpers.plot_correlations_with_parameters(correlations, pt_tempo_parameters)
        figname = 'correlations_dt{}dkmax{}esprel{}temp{}'.format(dt,dkmax,esprel,_temperature)
        figname_replaced = figname.replace('.','-')
        plt.savefig(figname_replaced)
        plt.show()

    process_tensor = tempo.pt_tempo_compute(bath=bath,
                                            start_time=0,
                                            end_time=max_time,
                                            parameters=pt_tempo_parameters,
                                            # progress_type='silent'
                                            )

    if name is None:
        name = 'optimisation_dt{}dkmax{}esprel{}temp{}maxtime{}'.format(dt,dkmax,esprel,_temperature,max_time)
        if alpha != 0.126:
            name = 'optimisation_dt{}dkmax{}esprel{}temp{}alpha{}maxtime{}'.format(dt,dkmax,esprel,_temperature,alpha,max_time)

        name_replaced = name.replace('.','-')
    process_tensor.export(qsl_path+name_replaced+".processTensor",overwrite=True)



def generate_transmon_PT(dt,
                dkmax,
                esprel,
                _temperature, # units of Kelvin
                max_time,
                # alpha = 0.126,
                name = None,
                plot_correlations=True,
                disable_coupling = False):
    # generate the process tensor and save to file
    omega_cutoff = 400
    if disable_coupling is True:
        alpha = 0.0

    temperature = _temperature #* 0.1309 # 1K = 0.1309/ps
    
    correlations = tempo.CustomSD(j_function=sd, cutoff=400, cutoff_type='hard', temperature=temperature)
    bath = tempo.Bath(tempo.operators.sigma("x"), correlations)

    pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=esprel)

    if plot_correlations:
        tempo.helpers.plot_correlations_with_parameters(correlations, pt_tempo_parameters)
        figname = 'correlations_dt{}dkmax{}esprel{}temp{}'.format(dt,dkmax,esprel,_temperature)
        figname_replaced = figname.replace('.','-')
        plt.savefig(figname_replaced)

        plt.show()

    process_tensor = tempo.pt_tempo_compute(bath=bath,
                                            start_time=0,
                                            end_time=max_time,
                                            parameters=pt_tempo_parameters,
                                            progress_type='bar'
                                            )
    if name is None:
        name = 'optimisation_dt{}dkmax{}esprel{}temp{}maxtime{}'.format(dt,dkmax,esprel,_temperature,max_time)

        name_replaced = name.replace('.','-')
    process_tensor.export(qsl_path+name_replaced+".processTensor",overwrite=True)



def generate_pure_dephasing_pt(dt,
                dissipator_rate,
                max_time,
            ) -> tempo.SimpleAdjointTensor:
    '''
    generates a process tensor that just consists of Liouvillian superoperators
    for pure dephasing
    '''

    tensor = np.identity(4)
    # factor of 2 here is to keep convention defined by Chin et al.
    exp_value = np.exp(-2 * dissipator_rate * dt)
    tensor[1,1] = exp_value
    tensor[2,2] = exp_value
    tensor = tensor[np.newaxis,:,:]
    tensor = tensor[np.newaxis,:,:,:]

    pt = tempo.SimpleAdjointTensor(
                    hilbert_space_dimension=2,
                    dt=dt)

    times = np.arange(0,max_time,dt)

    for i in range(times.size):
        pt.set_mpo_tensor(i,tensor)
        pt.set_cap_tensor(i,np.array([1]))
    pt.set_cap_tensor(times.size,np.array([1]))

    name = 'pure_dephasing_dissipator_rate_{}maxtime{}'.format(dissipator_rate,max_time)
    name_replaced = name.replace('.','-')
    pt.export(qsl_path+name_replaced+".processTensor",overwrite=True)
