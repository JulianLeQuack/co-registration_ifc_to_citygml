import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

import numpy as np
import matplotlib.pyplot as plt

from .rigid_transformation import Rigid_Transformation


def transform_IFC_horizontal(path_to_IFC: str, export_path: str, transformation: Rigid_Transformation):
    '''
    Applies the 2D rigid transformation to the input IFC file and stores it
    '''
    ifc_file = ifcopenshell.open(path_to_IFC)
        # Get settings for geometry processing
    settings = ifcopenshell.geom.settings()