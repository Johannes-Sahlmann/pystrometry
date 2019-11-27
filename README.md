[![Build Status](https://travis-ci.org/Johannes-Sahlmann/pystrometry.svg?branch=master)](https://travis-ci.org/Johannes-Sahlmann/pystrometry)
[![Documentation Status](https://readthedocs.org/projects/pystrometry/badge/?version=latest)](https://pystrometry.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pystrometry.svg)](https://badge.fury.io/py/pystrometry)
[![PyPI - License](https://img.shields.io/pypi/l/Django.svg)](https://github.com/Johannes-Sahlmann/pystrometry/blob/master/LICENSE.md)
[![DOI](https://zenodo.org/badge/172252669.svg)](https://zenodo.org/badge/latestdoi/172252669)

# pystrometry  -  Support for high-precision astrometry timeseries analysis


### Example usage
- Plotting the orbital motion with default parameters 


    from pystrometry.pystrometry import OrbitSystem 
    orb = OrbitSystem()  # default parameters
    orb.plot_orbits() 

- Define the orbital parameters


        attribute_dict = OrderedDict([  ('RA_deg', 164.), 
                                        ('DE_deg', -21.),
                                        ('absolute_plx_mas', 27.), 
                                        ('Tp_day', 57678.4), 
                                        ('omega_deg', -23.),
                                        ('P_day', 687.), 
                                        ('ecc', 0.08), 
                                        ('OMEGA_deg', 114.),
                                        ('i_deg', 31.), 
                                        ('m1_MS', 0.9),
                                        ('m2_MJ', 3.)])
                                        
        orb = OrbitSystem(attribute_dict)
        orb.plot_orbits() 




### Documentation


### Contributing
Please open a new issue or new pull request for bugs, feedback, or new features you would like to see. If there is an issue you would like to work on, please leave a comment and we will be happy to assist. New contributions and contributors are very welcome!   
 

### References



