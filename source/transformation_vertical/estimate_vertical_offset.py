import numpy as np

def estimate_vertical_offset(citygml_extents, ifc_extents) -> float:
    max_z_ifc = max(storey['max_z'] for storey in ifc_extents if storey['max_z'] is not None)
    min_z_ifc = min(storey['min_z'] for storey in ifc_extents if storey['min_z'] is not None)
    max_z_citygml = citygml_extents["max_z"]
    min_z_citygml = citygml_extents["min_z"]

    if -2 <= (max_z_ifc - min_z_ifc) - (max_z_citygml - min_z_citygml) <= 2:
        offset = (min_z_citygml + (max_z_citygml - min_z_citygml)/2) - (min_z_ifc + (max_z_ifc - min_z_ifc)/2)
        return offset
    else:
        print("Warning: The vertical extents of the IFC and CityGML files do not match closely enough.")
        return None