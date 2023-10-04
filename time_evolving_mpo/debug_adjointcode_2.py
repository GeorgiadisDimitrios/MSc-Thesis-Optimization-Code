# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import matplotlib.pyplot as plt
import tensornetwork as tn
from scipy.linalg import expm

import numpy as np
import time_evolving_mpo as tempo


def generate_pt():
    # generate the process tensor and save to file
    omega_cutoff = 3.04
    alpha = 0.126
    temperature = 0.1309 # 1K = 0.1309/ps

    correlations = tempo.PowerLawSD(alpha=alpha,
                                    zeta=3,
                                    cutoff=omega_cutoff,
                                    cutoff_type='gaussian',
                                    max_correlation_time=5.0,
                                    temperature=temperature)
    bath = tempo.Bath(tempo.operators.sigma("z")/2.0, correlations)

    pt_tempo_parameters = tempo.PtTempoParameters(dt=0.02, dkmax=20, epsrel=10**(-4))

    process_tensor = tempo.pt_tempo_compute(bath=bath,
                                            start_time=-2.0,
                                            end_time=3.0,
                                            parameters=pt_tempo_parameters)


    process_tensor.export("details_pt_tempo_dt0p02.processTensor",overwrite=True)

# generate_pt()


# %%
# import the process tensor as an adjoint pt and set up the system
pt = tempo.import_adjoint_process_tensor('details_pt_tempo_dt0p02.processTensor','adjoint')

delta=0.0

initial_state=tempo.operators.spin_dm("z-")

def gaussian_shape(t, area = 1.0, tau = 1.0, t_0 = 0.0):
    return area/(tau*np.sqrt(np.pi)) * np.exp(-(t-t_0)**2/(tau**2))

def hamiltonian_t(t, delta=delta):
        return delta/2.0 * tempo.operators.sigma("z") + gaussian_shape(t-2.5, area = np.pi/2.0, tau = 0.245)/2.0 * tempo.operators.sigma("x")

system = tempo.TimeDependentSystem(hamiltonian_t)


# %%
# compute the dynamics using normal PT-TEMPO code for comparison
dynamics = pt.compute_dynamics_from_system(
        system=system,
        initial_state=initial_state)
t, s_x = dynamics.expectations(tempo.operators.sigma("x"), real=True)
_, s_y = dynamics.expectations(tempo.operators.sigma("y"), real=True)
_, s_z = dynamics.expectations(tempo.operators.sigma("z"), real=True)
s_xy = np.sqrt(s_x**2 + s_y**2)


# %%

Omega_t = gaussian_shape(t-2.5, area = np.pi/2.0, tau = 0.245)
#plt.plot(t,Omega_t/4.0,label=r"Driving pulse")
def plot_dynamics():
    plt.rc('font',size=18)
    plt.figure(1,figsize=(10,6))
    plt.fill_between(t, Omega_t/4.0, 0,
                    facecolor="orange", # The fill color
                    color='blue',       # The outline color
                    alpha=0.2,label=r"Driving pulse")

    plt.plot(t,s_z,label=r"$\langle\sigma_z\rangle$")
    plt.plot(t,s_y,label=r"$\langle\sigma_y\rangle$")
    plt.plot(t,s_x,label=r"$\langle\sigma_x\rangle$")
    plt.xlabel(r"$t\,/\mathrm{ps}$")
    #plt.ylabel(r"$\mathrm{ps}^{-1}$")
    plt.legend()
    plt.savefig("test2.pdf")


# %%
# compute the dynamics and the gradients with respect to the Liouville-space propagators
# using the adjoint code
derivlist = pt.compute_derivatives_from_system(system,initial_state=np.array([[0,0],[0,1]]),target_state = tempo.operators.spin_dm("y+"))





# %%
times = 0.0 + np.arange(len(pt.forwards)/2)*pt.dt


# %%
import itertools


# %%
# to convert the derivatives w.r.t. the propagators into derivatives with respect to a term in Hamiltonian
# we use the fact that dF/dr= sum a,b dF/dU_a,b dU_a,b/dr.
# so I need the derivatives of the U (system propagators in Liouville space) w.r.t. control parameter
# Here is a simple centred-finite difference calculation, with step h
# for the case where we want dF/dr with r the prefactor of op in the Hamiltonian
def dpropagator(system,t,dt,op,h):
    temp=h
    h=temp
    ham=system.hamiltonian
    post_liouvillian_plush=-1j * tempo.util.commutator(ham(t+dt*3.0/4.0)+h*op)
    post_liouvillian_minush=-1j * tempo.util.commutator(ham(t+dt*3.0/4.0)-h*op)

    post_propagator_plush=expm(post_liouvillian_plush*dt/2.0).T
    post_propagator_minush=expm(post_liouvillian_minush*dt/2.0).T

    postderiv=(post_propagator_plush-post_propagator_minush)/(2.*h)

    pre_liouvillian_plush=-1j * tempo.util.commutator(ham(t+dt*1.0/4.0)+h*op)
    pre_liouvillian_minush=-1j * tempo.util.commutator(ham(t+dt*1.0/4.0)-h*op)

    pre_propagator_plush=expm(pre_liouvillian_plush*dt/2.0).T
    pre_propagator_minush=expm(pre_liouvillian_minush*dt/2.0).T
    prederiv=(pre_propagator_plush-pre_propagator_minush)/(2.*h)
    return prederiv,postderiv

# construct the derivatives of propagators with
def dpropagatorlist(system,times,dt,op,h=1e-6):
    derivlist=[]
    for step in range(len(times)):
        prederiv,postderiv=dpropagator(system,times[step],dt,op,h)
        derivlist.append(prederiv)
        derivlist.append(postderiv)
    return derivlist

def combinederivs(target_derivatives,propagator_derivatives):
    assert (len(target_derivatives)==len(propagator_derivatives)), "Lists supplied have uneqal length"
    derivvalslist=[]
    for (target_deriv,propagator_deriv) in zip(target_derivatives,propagator_derivatives):
        # I need to check I've got these indices the right way
        # here's the tensornetwork version
        #dtargetnode=tn.Node(target_deriv)
        #dpropnode=tn.Node(propagator_deriv)
        #dtargetnode[0] ^ dpropnode[0]
        #dtargetnode[1] ^ dpropnode[1]
        #derivvalslist.append((dtargetnode @ dpropnode).tensor+0.0)
        # if the above is right it can also be done like this
        derivvalslist.append(np.matmul(target_deriv.T,propagator_deriv).trace())
    return derivvalslist

def allproptimes(times,dt):
    return np.array([[t+dt/4.0,t+dt*3.0/4.0] for t in times]).flatten()


# %%
# functions to evaluate the density matrix at the half steps
# probably destructive of the tensor networks contained in proctens.forwards,

def densmatlist(proctens):
    fplist=proctens.forwards
    resultlist=[]
    print(len(fplist)//2)
    cap=tn.Node(proctens.get_cap_tensor(0))
    for i in range(len(fplist)//2):
        curr=fplist[2*i]
        curr[0] ^ cap[0]
        result=curr @ cap
        resultlist.append(result.tensor.reshape((2,2)))
        cap=tn.Node(proctens.get_cap_tensor(i+1))
        curr=fplist[2*i+1]
        curr[0] ^ cap[0]
        result=curr @ cap
        resultlist.append(result.tensor.reshape((2,2)))
    return resultlist

def plot_derivatives():
    # %%
    dpropsz=dpropagatorlist(system,times,pt.dt,np.array([[1.0,0.0],[0.0,-1.0]]))
    dpropsx=dpropagatorlist(system,times,pt.dt,np.array([[0.0,1.0],[1.0,0.0]]))
    dpropsy=dpropagatorlist(system,times,pt.dt,np.array([[0.0,-1.0j],[1.0j,0.0]]))


    # %%
    szderivs=combinederivs(dpropsz,derivlist)
    sxderivs=combinederivs(dpropsx,derivlist)
    syderivs=combinederivs(dpropsy,derivlist)


    # %%
    preposttimes=allproptimes(times,pt.dt)


    # %%

    Omega_t = gaussian_shape(t-2.5, area = np.pi/2.0, tau = 0.245)
    #plt.plot(t,Omega_t/4.0,label=r"Driving pulse")

    plt.rc('font',size=18)
    plt.figure(figsize=(10,6))
    plt.fill_between(t, Omega_t/4000.0, 0,
                    facecolor="orange", # The fill color
                    color='blue',       # The outline color
                    alpha=0.2,label=r"Driving pulse")

    plt.plot(preposttimes,szderivs,label=r"z")
    np.save('times', preposttimes)
    np.save('szderivs',szderivs)
    np.save('sxderivs',sxderivs)
    np.save('syderivs',syderivs)
    plt.plot(preposttimes,syderivs,label=r"y")
    plt.plot(preposttimes,sxderivs,label=r"x")
    plt.xlabel(r"$t\,/\mathrm{ps}$")
    plt.ylabel(r"$dF/dh_i\;/\mathrm{ps}$")
    plt.legend()
    plt.savefig("test.png",format="png")


    # %%
    # plot the expectations on the half-steps
    # this might be destructive of the data stored in pt.forwards
    rdm=densmatlist(pt)
    op=np.array([[1.0,0.0],[0.0,-1.0]])
    szexps=list(map(lambda x: (np.matmul(x,op).trace()),rdm))
    op=np.array([[0.0,-1.0j],[1.0j,0.0]])
    syexps=list(map(lambda x: (np.matmul(x,op).trace()),rdm))
    op=np.array([[0.0,1.0],[1.0,0.0]])
    sxexps=list(map(lambda x: (np.matmul(x,op).trace()),rdm))

    plt.rc('font',size=18)
    plt.figure(figsize=(10,6))
    plt.plot(preposttimes,sxexps)
    plt.plot(preposttimes,syexps)
    plt.plot(preposttimes,szexps)

plot_dynamics()
plot_derivatives()
plt.show()


def this_broken_rn():
    # %%
    # hmm. that looks good but here are some checks. what happens if we turn off dissipation
    pt = tempo.import_adjoint_process_tensor('details_pt_tempo_nobath.processTensor','adjoint')


    # %%
    # compute the dynamics and the gradients with respect to the Liouville-space propagators
    # using the adjoint code
    derivlist,final_state,_ = pt.compute_derivatives_from_system(system,initial_state=np.array([[0,0],[0,1]]),target_state = np.array([[1,0],[0,0]]))


    # %%



    # %%
    szderivs=combinederivs(dpropsz,derivlist)
    sxderivs=combinederivs(dpropsx,derivlist)
    syderivs=combinederivs(dpropsy,derivlist)


    # %%



    # %%
    plt.plot(preposttimes,szderivs)
    plt.plot(preposttimes,sxderivs)
    plt.plot(preposttimes,syderivs)


    # %%
    # excellent, that looks comparable to the numerical accuracy (given h=1e-6 particularly)


    # %%
    (np.finfo(float).eps)**(1.0/3.0) # ah ok so if i compute to a fractional accuracy of machine precision, then 6e-6 times the scale of the function is the optimal choice of h


    # %%
    a=3.017484
    temp=a
    a=temp
    print(a)


    # %%
    ((pt.dt)**2)**(1.0/3.0) # so don't compute the propagator approximately! this would be the optimal h otherwise.


    # %%
    tempo.operators.spin_dm("x+")


    # %%
    # ok, back to the original thing but let's calculate the gradient wrt the fidelity to a different state
    derivlist,final_state = pt.compute_derivatives_from_system(system,initial_state=np.array([[0,0],[0,1]]),target_state = tempo.operators.spin_dm("y+"))


    # %%
    szderivs=combinederivs(dpropsz,derivlist)
    sxderivs=combinederivs(dpropsx,derivlist)
    syderivs=combinederivs(dpropsy,derivlist)
    plt.plot(preposttimes,szderivs,'r-')
    plt.plot(preposttimes,sxderivs,'g-')
    plt.plot(preposttimes,syderivs,'b-')


    # %%
    get_ipython().run_line_magic('pinfo', 'plt.plot')


    # %%



    # %%
    szderivs=combinederivs(dpropsz,derivlist)
    sxderivs=combinederivs(dpropsx,derivlist)
    syderivs=combinederivs(dpropsy,derivlist)
    plt.plot(preposttimes,szderivs,'r-')
    plt.plot(preposttimes,sxderivs,'g-')
    plt.plot(preposttimes,syderivs,'b-')


# # %%
# # those results look good. i wonder if i can check by finite differencing?
# # no that's too boring?
# # perhaps not.


# # %%
# pt.forwards[]


# # %%
# pt.get_cap_tensor(1)


# # %%



# # %%
# (0+1)//2


# # %%
# (1+1)//2


# # %%



# # %%



# # %%

# #plt.plot(t,s_y)


# # %%
# rd


# # %%
# preposttimes.shape


# # %%
# pt.dt


# # %%
# pt.forwards


# # %%



