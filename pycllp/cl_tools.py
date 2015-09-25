"""
Various helper functions and attributes for working with pyopencl and CL code.
"""
import pyopencl as cl
# Convenient path OpenCL code
import os
CL_PATH = os.path.dirname(__file__)
CL_PATH = os.path.join(CL_PATH, 'cl')


def cl_program_from_file(context, filename):
    """
    Returns a pyopencl.Program with the source from filename.

    Filename should be present in the CL_PATH folder.
    """
    return cl.Program(context, open(os.path.join(CL_PATH, filename)).read())
