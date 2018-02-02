"""
OpenMDAO Wrapper for ParOpt -- a parallel interior point optimizer.

ParOpt is a parallel optimizer for use in general large-scale
optimization applications, but is often specifically used for topology
and multi-material optimization problems. The optimizer has the
capability to handle large numbers of weighting constraints that arise
in the parametrization of multi-material problems. The implementation
of the optimizer is in C++ and uses MPI.
"""

from __future__ import print_function
from collections import OrderedDict
import traceback

from six import iteritems

import numpy as np
from scipy.sparse import coo_matrix

#from pyoptsparse import Optimization
from paropt import ParOpt

from openmdao.core.analysis_error import AnalysisError
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.record_util import create_local_meta

# names of optimizers that use gradients
grad_drivers = {'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
                'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG',
                'PAROPT'}

# names of optimizers that allow multiple objectives
multi_obj_drivers = {'NSGA2'}

def _check_imports():
    """
    Dynamically remove optimizers we don't have.

    Returns
    -------
    list of str
        List of valid optimizer strings.
    """
    optlist = ['ALPSO', 'CONMIN', 'FSQP', 'IPOPT', 'NLPQLP',
               'NSGA2', 'PSQP', 'SLSQP', 'SNOPT', 'NLPY_AUGLAG',
               'NOMAD', 'PAROPT']

    for optimizer in optlist[:]:
        try:
            __import__('paropt', globals(), locals(), [optimizer], 0)
        except ImportError:
            optlist.remove(optimizer)

    return optlist

CITATIONS = """
@inproceedings{Kennedy:2015:SciTech,
title={Large-Scale Multimaterial Topology Optimization for Additive Manufacturing},
author={Graeme J. Kennedy},
year={2015},
booktitle = {56th AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference},
month = {January},
doi={10.2514/6.2015-1799},
address = {Kissimmee, {FL}}}
"""

class ParOptSparseDriver(Driver):
    """
    Driver wrapper for ParOpt.

    #TODO: implement the following functions in the context of ParOpt
    """
    def __init__(self):
        """
        Initialize Paropt.
        """
        return

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        return

    def _objfunc(self, x_new):
        """
        Evaluate and return the objective and constraint function.

        Model is executed here.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        return
        
    def _gradfunc(self, x_new):
        """
        Evaluate and return the gradient for the objective.

        Gradients for the constraints are also calculated and cached here.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        ndarray
            Gradient of objective with respect to parameter array.
        """
        return

    def _get_name(self):
        """
        Get name of current driver.

        Returns
        -------
        optimizer : str
            The name of the current driver.
        """
        return self.options['optimizer']

    def _setup_simul_coloring(self, mode='fwd'):
        """
        Set up metadata for simultaneous derivative solution.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        """
        return
    
    def run(self):
        """
        Excute pyOptsparse.
        
        Note that pyOpt controls the execution, and the individual optimizers
        (e.g., SNOPT) control the iteration.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        return
