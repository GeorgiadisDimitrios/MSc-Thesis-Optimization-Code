'''
Methods used during optimisation
'''
import numpy as np
import time_evolving_mpo as tempo
from path import qsl_path

def load_process_tensor(max_time: float,
                        dt: float,
                        dkmax: float,
                        esprel: float,
                        temperature: float)-> tempo.SimpleAdjointTensor:
    '''
    Given parameters and max time, import process tensor as adjoint PT
    '''



    name = 'optimisation_dt{}dkmax{}esprel{}temp{}maxtime{}'.format(dt,dkmax,esprel,temperature,max_time)


    name_replaced = name.replace('.','-')
    pt = tempo.import_adjoint_process_tensor(qsl_path+name_replaced+'.processTensor','adjoint')
    return pt
