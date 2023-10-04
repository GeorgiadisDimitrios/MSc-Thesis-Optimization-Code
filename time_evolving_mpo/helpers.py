# Copyright 2021 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file  in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Handy helper functions.
"""

from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from time_evolving_mpo.correlations import BaseCorrelations
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.tempo import TempoParameters
from time_evolving_mpo.operators import sigma




def plot_correlations_with_parameters(
        correlations: BaseCorrelations,
        parameters: TempoParameters,
        ax: Axes = None) -> Axes:
    """Plot the correlation function on a grid that corresponds to some
    tempo parameters. For comparison, it also draws a solid line that is 10%
    longer and has two more sampling points per interval.

    Parameters
    ----------
    correlations: BaseCorrelations
        The correlation obeject we are interested in.
    parameters: TempoParameters
        The tempo parameters that determine the grid.
    """
    times = parameters.dt/3.0 * np.arange(int(parameters.dkmax*3.3))
    corr_func = np.vectorize(correlations.correlation)
    corr_vals = corr_func(times)
    sample = [3*i for i in range(parameters.dkmax)]

    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$C(\tau)$")
    ax.plot(times, np.real(corr_vals), color="C0", linestyle="-", label="real")
    ax.scatter(times[sample], np.real(corr_vals[sample]), marker="d", color="C0")
    ax.plot(times, np.imag(corr_vals), color="C1", linestyle="-", label="imag")
    ax.scatter(times[sample], np.imag(corr_vals[sample]), marker="o", color="C1")
    ax.legend()
    ax.grid()

    if show:
        fig.show()
    return ax

def plot_bloch_sphere(
                    dynamics: Dynamics,
                    update_rc = False,
                    show = False,
                    save = False,
                    filename = None
                    ):
    '''
    Takes given tempo.dynamics and plots them on the bloch sphere
    '''
    assert isinstance(dynamics,Dynamics), 'Dynamics must be of type tempo.dynamics'

    if update_rc:
        from matplotlib import rc

        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['text.usetex'] = True

    # plots circle that is the bloch sphere
    number_of_points = 1000

    theta_array = np.linspace(0,np.pi,number_of_points)
    phi_array = np.linspace(0,2 * np.pi,number_of_points)

    # radius of the bloch sphere
    r = 1 # hmmm, i suppose this could be 1/2

    x = r * np.outer(np.cos(phi_array),np.sin(theta_array))
    y = r * np.outer(np.sin(phi_array),np.sin(theta_array))
    z = r * np.outer(np.ones(number_of_points),np.cos(theta_array))


    # Plot
    cm = 1/2.54  # centimeters in inches
    width = 16
    aspect_ratio = 2/3
    # aspect_ratio = 9/16
    # aspect_ratio = 1
    height = width * aspect_ratio



    # fig = plt.figure(figsize=(18,12))
    # fig = plt.figure(figsize=(18,12))
    fig = plt.figure(figsize=(width*cm,height*cm))

    ax = fig.add_subplot(projection='3d')


    t,sx = dynamics.expectations(sigma('x'),real=True)
    t,sy = dynamics.expectations(sigma('y'),real=True)
    t,sz = dynamics.expectations(sigma('z'),real=True)

    magnitude = np.sqrt(sx**2+sy**2+sz**2)
    magnitude_rescaled = (magnitude - magnitude.min()) / (magnitude.max()-magnitude.min())
    total_time_frac = t / t.max()

    colourmap = plt.cm.get_cmap('viridis')

    # i sorta kinda stole this from
    # https://stackoverflow.com/questions/23810274/python-colormap-in-matplotlib-for-3d-line-plot
    for i in range(t.size):
        ax.plot(sx[i],sy[i],sz[i],marker='o',color=colourmap(magnitude_rescaled[i]))
    # ax.plot(sx,sy,sz)

    a_range = np.linspace(-1,1,50)
    zeros = np.zeros(a_range.size)

    #the axes
    ax.plot(a_range,zeros,zeros,color='black') # x
    ax.plot(zeros,a_range,zeros,color='black') # y
    ax.plot(zeros,zeros,a_range,color='black') # z

    # add wireframe to make bloch sphere surface
    ax.plot_wireframe(x,y,z,rcount=5,ccount=5,color='lightgrey') # rcount and ccount give the number of lines


    ax.text(0,0,-1.1,r'$|z^-\rangle$')
    ax.text(0,0,1.1,r'$|z^+\rangle$')
    ax.text(0,-1.1,0,r'$|y^-\rangle$')
    ax.text(0,1.1,0,r'$|y^+\rangle$')
    ax.text(-1.1,0,0,r'$|x^-\rangle$')
    ax.text(1.1,0,0,r'$|x^+\rangle$')

    ax.plot(0,0,-1,'bx')
    ax.plot(0,0,1,'bx')
    ax.plot(0,-1,0,'bx')
    ax.plot(0,1,0,'bx')
    ax.plot(-1,0,0,'bx')
    ax.plot(1,0,0,'bx')

    ax.set_xlabel(r"$\sigma_x$")
    ax.set_ylabel(r"$\sigma_y$")
    ax.set_zlabel(r"$\sigma_z$")

    # https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
    ticks = np.array([-1,-0.5,0,0.5,1])
    ax.xaxis.set_ticks(ticks)
    ax.yaxis.set_ticks(ticks)
    ax.zaxis.set_ticks(ticks)


    ax.view_init(elev=23,azim=-10)
    norm = Normalize(vmin=magnitude.min(),vmax=magnitude.max())
    ticks = np.arange(magnitude.min(),magnitude.max(),10)
    cax = plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=colourmap),ax=ax)
    cax.ticks = ticks

    if save:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.savefig('bloch_sphere.pdf')
    plt.tight_layout()


    #bar scale for bloch vector magnitude
    # plt.figure(2)
    # line = np.linspace(0,1,100)
    # plane = np.vstack(line,line)
    # plt.imshow(line,extent=(magnitude.min(),magnitude.max()))

    if show:
        plt.show()


def plot_bloch_sphere_for_slideshow(
                    dynamics: Dynamics,
                    update_rc = False,
                    fig = None,
                    show = False,
                    save = False,
                    filename = None
                    ):
    '''
    Takes given tempo.dynamics and plots them on the bloch sphere
    '''
    assert isinstance(dynamics,Dynamics), 'Dynamics must be of type tempo.dynamics'

    if update_rc:
        from matplotlib import rc

        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['text.usetex'] = True

    # plots circle that is the bloch sphere
    number_of_points = 1000

    theta_array = np.linspace(0,np.pi,number_of_points)
    phi_array = np.linspace(0,2 * np.pi,number_of_points)

    # radius of the bloch sphere
    r = 1 # hmmm, i suppose this could be 1/2

    x = r * np.outer(np.cos(phi_array),np.sin(theta_array))
    y = r * np.outer(np.sin(phi_array),np.sin(theta_array))
    z = r * np.outer(np.ones(number_of_points),np.cos(theta_array))


    # Plot
    if fig is not None:
        ax = fig.add_subplot(2,2,4,projection='3d')


    else:
        fig = plt.figure(figsize=(18,12))
        ax = fig.add_subplot(projection='3d')



    ax.plot_wireframe(x,y,z,rcount=5,ccount=5,color='lightgrey') # rcount and ccount give the number of lines

    t,sx = dynamics.expectations(sigma('x'),real=True)
    t,sy = dynamics.expectations(sigma('y'),real=True)
    t,sz = dynamics.expectations(sigma('z'),real=True)

    magnitude = np.sqrt(sx**2+sy**2+sz**2)
    magnitude_rescaled = (magnitude - magnitude.min()) / (magnitude.max()-magnitude.min())
    total_time_frac = t / t.max()

    colourmap = plt.cm.get_cmap('viridis')

    # i sorta kinda stole this from
    # https://stackoverflow.com/questions/23810274/python-colormap-in-matplotlib-for-3d-line-plot
    for i in range(t.size):
        ax.plot(sx[i],sy[i],sz[i],marker='o',color=colourmap(magnitude_rescaled[i]))
    # ax.plot(sx,sy,sz)

    a_range = np.linspace(-1,1,50)
    zeros = np.zeros(a_range.size)

    #the axes
    ax.plot(a_range,zeros,zeros,color='black') # x
    ax.plot(zeros,a_range,zeros,color='black') # y
    ax.plot(zeros,zeros,a_range,color='black') # z

    ax.text(0,0,-1.1,r'$|z^-\rangle$')
    ax.text(0,0,1.1,r'$|z^+\rangle$')
    ax.text(0,-1.1,0,r'$|y^-\rangle$')
    ax.text(0,1.1,0,r'$|y^+\rangle$')
    ax.text(-1.1,0,0,r'$|x^-\rangle$')
    ax.text(1.1,0,0,r'$|x^+\rangle$')

    ax.plot(0,0,-1,'bx')
    ax.plot(0,0,1,'bx')
    ax.plot(0,-1,0,'bx')
    ax.plot(0,1,0,'bx')
    ax.plot(-1,0,0,'bx')
    ax.plot(1,0,0,'bx')
    ax.set_xlabel(r"$\sigma_x$")
    ax.set_ylabel(r"$\sigma_y$")
    ax.set_zlabel(r"$\sigma_z$")

    # https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-x-or-y-axis-in-matplotlib
    ticks = np.array([-1,-0.5,0,0.5,1])
    ax.xaxis.set_ticks(ticks)
    ax.yaxis.set_ticks(ticks)
    ax.zaxis.set_ticks(ticks)


    ax.view_init(elev=23,azim=-10)
    norm = Normalize(vmin=magnitude.min(),vmax=1)
    ticks = np.arange(magnitude.min(),1,10)
    cax = plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=colourmap),ax=ax)
    cax.ticks = ticks

    if save:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.savefig('bloch_sphere.pdf')
    plt.tight_layout()


    #bar scale for bloch vector magnitude
    # plt.figure(2)
    # line = np.linspace(0,1,100)
    # plane = np.vstack(line,line)
    # plt.imshow(line,extent=(magnitude.min(),magnitude.max()))

    if show:
        plt.show()