import os
from time import time




import time_evolving_mpo as tempo
import numpy as np
import matplotlib.pyplot as plt

from scipy import constants

from time import time

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True


Omega = 0.0 # 1.0
# omega_cutoff = 3.0
# alpha = 0.1
omega_cutoff = 2.0e-3* constants.electron_volt * 1e-12 / constants.hbar
alpha = constants.hbar * 11.2e-3 * omega_cutoff**2 * 1e12/ \
        (2 * constants.pi * constants.Boltzmann)

a = 0
tau_0 = 1.0
theta_0 = np.pi

start_time = -5
end_time = 5

dt = 0.1 # 0.01
dkmax = 20 # 200
epsrel = 1.0e-6# 1.0e-7

def delta_function(t):
    delta = - a * t / (a**2 + tau_0**4)
    return delta

def omega_function(t):
    omega  = theta_0 / (np.sqrt(2*np.pi)*np.sqrt(a**2 + tau_0**4)) * np.exp(- (t**2 * tau_0**2) / (2 * (a**2 + tau_0**4)))
    return omega

def system_hamiltonian(t):
    return 0.5 * delta_function(t) * tempo.operators.sigma('z') - 0.5 * omega_function(t) * tempo.operators.sigma("x")

system = tempo.AdjointTimeDependentSystem(system_hamiltonian)
initial_state = tempo.operators.spin_dm("down")
target_state = tempo.operators.spin_dm("up")
def generate_pt(temp):
    temp_new_units = temp * constants.Boltzmann * 1e-12 / constants.hbar


    correlations = tempo.PowerLawSD(alpha=alpha,
                                    zeta=3,
                                    cutoff=omega_cutoff,
                                    cutoff_type='gaussian',
                                    temperature=temp_new_units)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)

    # tempo_parameters = tempo.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

    pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

    pt = tempo.pt_tempo_compute(bath=bath,
                                start_time=start_time,
                                end_time=end_time,
                                parameters=pt_tempo_parameters,
                                progress_type='bar')
    pt.export("details_pt_tempo_{}K.processTensor".format(temp),overwrite=True)

def generate_lots_of_PTs(temp_list):
    for i in temp_list:
        generate_pt(i)


def plot_lots_of_PTs(temp_list):
    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    for i,temp in enumerate(temp_list):
        plt.figure(0)
        pt = tempo.import_adjoint_process_tensor("details_pt_tempo_{}K.processTensor".format(temp),'adjoint')

        st = time()
        dyn = pt.compute_dynamics_from_system(system=system, initial_state=initial_state,start_time=start_time)
        et = time()
        print('Time = {}'.format(et-st))


        t2, s2_y = dyn.expectations(0.5 * tempo.operators.sigma("z"), real=True)

        plt.plot(t2, s2_y, label=r'{}K'.format(temp),color=colours[i],zorder=6-i)

        plt.xlabel(r'$t(ps)$')
        plt.ylabel(r'$\langle \sigma_z\rangle$')
        plt.ylim(-0.5,0.5)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.figure(1)
        time1 = time()
        pt.compute_derivatives_from_system(system,initial_state=initial_state,target_state=target_state,start_time=start_time)
        time2 = time()
        result = pt.compute_chain_rule(system,derivatives_array,start_time=-5.0)
        time3 = time()


        print('time taken to compute adjoint tensor = {}'.format(time2-time1))
        print('time taken to compute derivatives = {}'.format(time3-time2))


        plt.plot(x_time,result,label=r'{}K'.format(temp),color=colours[i],zorder=6-i)
        plt.xlabel(r'$t(ps)$')
        plt.ylabel(r'$\frac{\partial F}{\partial \Omega(t)}$')
        plt.grid()
        plt.legend()
        plt.tight_layout()





    # def analytic_solution(t):
    #     x = (t*omega_cutoff)**2
    #     phi = 2 * alpha * (1 + (x-1)/(x+1)**2)
    #     return np.exp(-phi)


    # sy_exact = analytic_solution(t2)
    # #plt.plot(t2, sy_exact, label=r'gerlalds exact', linestyle="dotted")




def plot_omega(start_time,end_time):
    plt.figure(2)
    x_time = np.linspace(start_time,end_time,100)
    plt.plot(x_time,omega_function(x_time))
    plt.xlabel(r't(ps)')
    plt.ylabel(r'$\Omega(t)\quad [ps^{-1}]$')
    plt.grid()
    plt.tight_layout()

def plot_delta(start_time,end_time):
    plt.figure(3)
    x_time = np.linspace(start_time,end_time,100)
    plt.plot(x_time,delta_function(x_time))
    plt.xlabel(r't(ps)')
    plt.ylabel(r'$\Delta(t)\quad [ps^{-1}]$')
    plt.grid()
    plt.tight_layout()

number_of_timesteps = int((end_time - start_time)/dt)
derivatives_array = np.zeros((number_of_timesteps,2,2),dtype=np.complex128)

x_time = np.arange(start_time,end_time,dt)

for i, t in enumerate(x_time):
    derivatives_array[i,:,:] = 0.5 * delta_function(t) * tempo.operators.sigma('z') - 0.5 * omega_function(t) * tempo.operators.sigma('x')/ theta_0


temp_list  = np.array([0.0001,5,20])
# temp_list = np.array([0.0001,0.01,1,3,5])
# temp_list = np.array([0.0001])

# generate_lots_of_PTs(temp_list)
plot_lots_of_PTs(temp_list)
plot_omega(start_time,end_time)
plot_delta(start_time,end_time)


plt.show()