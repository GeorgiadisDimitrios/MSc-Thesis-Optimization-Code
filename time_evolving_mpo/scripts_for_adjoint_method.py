
import sys
import numpy as np
import time_evolving_mpo as tempo


Omega = 1.0
omega_cutoff = 3.0
alpha = 0.1
temperature = 0.0
start_time = 0.0
end_time = 10.0
initial_state = tempo.operators.spin_dm("y+")

def generate_process_tensor():


    correlations = tempo.PowerLawSD(alpha=alpha,
                                    zeta=3,
                                    cutoff=omega_cutoff,
                                    cutoff_type='exponential',
                                    temperature=temperature)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)

    dt = 2.5 # 0.01
    dkmax = 3 # 200
    epsrel = 1.0e-6# 1.0e-7

    pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)
    pt = tempo.pt_tempo_compute(bath=bath,
                                start_time=start_time,
                                end_time=end_time,
                                parameters=pt_tempo_parameters,
                                progress_type='bar')
    pt.export("4steps.processTensor",overwrite=True)

def system_hamiltonian(t):
    return 1.0 * tempo.operators.sigma("x")

derivatives = np.array([tempo.operators.sigma("x"),tempo.operators.sigma("z")])

# generate_process_tensor()

pt = tempo.import_adjoint_process_tensor('4steps.processTensor','adjoint')
system = tempo.AdjointTimeDependentSystem(system_hamiltonian)
pt.compute_derivatives_from_system(system,initial_state=np.array([[1,0],[0,0]]),target_state = np.array([[0,0],[0,1]]))
result = pt.compute_chain_rule(system,derivatives,0)
print(result)

