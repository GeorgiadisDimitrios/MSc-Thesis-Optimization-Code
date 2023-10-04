## How to run
First you need to install the dependencies from the requirements.txt file.
Finally before running the script dimitrios_optimization.py you need to edit the base_path variable in
the paths.py file to the path of your local machine.

## Credits
Based on the work of Eoin P.Butler from Trinity College Dublin.

# Adjoint TimeEvolvingMPO
**Extending the TimeEvolvingMPO to have the adjoint method built in.**

This open source project aims to facilitate versatile numerical tools to efficiently compute the dynamics of quantum systems that are possibly strongly coupled to a structured environment. It allows to conveniently apply the so called time evolving matrix product operator method (TEMPO) [1], as well as the process tensor TEMPO method (PT-TEMPO) [2].

- **[1]**
A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).
- **[2]** G. E. Fux, E. Butler, P. R. Eastham, B. W. Lovett, and
J. Keeling, *Efficient exploration of Hamiltonian parameter space for
optimal control of non-Markovian open quantum systems*, 
Phys. Rev. Lett. 126, 200401(2021).

## Links

* Github:         <https://github.com/tempoCollaboration/TimeEvolvingMPO>
* Documentation:  <https://TimeEvolvingMPO.readthedocs.io>
* PyPI:           <https://pypi.org/project/time-evolving-mpo/>
* Tutorial:       [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tempoCollaboration/TimeEvolvingMPO/master?filepath=tutorials%2Ftutorial_01_quickstart.ipynb)


## Citing, Authors and Bibliography
See the files
[`HOW_TO_CITE.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/HOW_TO_CITE.md),
[`AUTHORS.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/AUTHORS.md)
and
[`BIBLIOGRAPHY.md`](https://github.com/tempoCollaboration/TimeEvolvingMPO/blob/master/BIBLIOGRAPHY.md).
