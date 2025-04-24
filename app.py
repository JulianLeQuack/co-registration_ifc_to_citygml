import os, shutil, tempfile, streamlit as st
from pathlib import Path
import sys

# define one temp folder
TMP_DIR = Path(tempfile.gettempdir()) / "bim2city_tmp"

# only reset on first run of this session
if "tmp_dir_initialized" not in st.session_state:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir()
    st.session_state.tmp_dir_initialized = True

import matplotlib.pyplot as plt

# Import your footprint and registration functions.
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint

from source.transformation_horizontal.detect_features import detect_features, filter_features_by_triangle_area
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation


# Page setup
st.title("bim2city")
# Sidebar navigation
page = st.sidebar.radio("Navigation", 
                          ["Input Data", 
                           "Footprint Creation Parameters", 
                           "Corner Detection & Filtering", 
                           "Rigid Registration Estimation"])

if page == "Input Data":
    st.header("Input Data")
    ifc_file = st.file_uploader("Upload IFC file", type=["ifc"])
    citygml_file = st.file_uploader("Upload CityGML file", type=["gml"])
    dxf_file = st.file_uploader("Upload DXF file", type=["dxf"])
    if ifc_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc", dir=str(TMP_DIR)) as tmp:
            tmp.write(ifc_file.getvalue())
            st.session_state.ifc_path = tmp.name
        st.success(f"IFC uploaded → temp path saved:\n{st.session_state.ifc_path}")
    if citygml_file is not None:
        st.write("CityGML file uploaded:", citygml_file.name)
    if dxf_file is not None:
        st.write("DXF file uploaded:", dxf_file.name)
        
elif page == "Footprint Creation Parameters":
    st.header("Footprint Creation Parameters")
    if "ifc_path" not in st.session_state:
        st.warning("Please upload an IFC file on the Input Data page first.")
    else:
        # remember last choice in session_state so changing re‑triggers rerun
        ifc_type = st.selectbox(
            "Select IFC Type", 
            ["IfcSlab", "IfcWall"], 
            key="ifc_type"
        )
        # always regenerate on select change
        with st.spinner(f"Generating footprint for {ifc_type}…"):
            footprint = create_IFC_footprint_polygon(
                ifc_path=st.session_state.ifc_path,
                ifc_type=ifc_type
            )
        if footprint:
            st.session_state.ifc_footprint = footprint
            fig, ax = plt.subplots(figsize=(6,6))
            for poly in footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="green", lw=2)
            ax.set_title(f"IFC Footprint ({ifc_type})")
            ax.set_aspect("equal", "box")
            st.pyplot(fig)
        else:
            st.error(f"No geometry returned for IFC type “{ifc_type}”.")
    
elif page == "Corner Detection & Filtering":
    st.header("Corner Detection & Filtering")
    if "ifc_footprint" not in st.session_state:
        st.warning("Please create an IFC footprint on the Footprint Creation Parameters page first.")
    else:
        footprint = st.session_state.ifc_footprint

        # sliders with keys so Streamlit tracks them
        angle_threshold = st.slider(
            "Corner Detection Angle Threshold (deg)", 
            0, 180, 30, key="angle_threshold"
        )
        min_area = st.slider(
            "Filter Triangle Area", 
            0.0, 100.0, 15.0, key="min_area"
        )

        # whenever sliders change, Streamlit reruns here
        with st.spinner("Detecting and filtering features…"):
            detected = detect_features(footprint, angle_threshold_deg=angle_threshold)
            filtered = filter_features_by_triangle_area(detected, footprint, min_area)

        fig, ax = plt.subplots(figsize=(6,6))
        # draw footprint
        for poly in footprint.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="blue", lw=1)

        # draw detected
        if detected.size:
            ax.scatter(
                detected[:,2], detected[:,3],
                color="orange", label=f"Detected ({len(detected)})"
            )
        # draw filtered
        if filtered.size:
            ax.scatter(
                filtered[:,2], filtered[:,3],
                color="red", label=f"Filtered ({len(filtered)})"
            )

        ax.set_title(f"Features @ angle≥{angle_threshold}° & area≥{min_area}")
        ax.set_aspect("equal", "box")
        ax.legend()
        st.pyplot(fig)
    
elif page == "Rigid Registration Estimation":
    st.header("Rigid Registration Estimation")
    distance_tol = st.number_input("Distance tolerance", min_value=0.0, value=1.0, step=0.1)
    angle_tol = st.number_input("Angle tolerance (deg)", min_value=0, value=45, step=1)
    if st.button("Run Registration"):
        st.write("Running rigid transformation estimation...")
        # In an actual implementation you would:
        # 1. Create footprints from the input data (IFC, DXF, CityGML)
        # 2. Extract and filter features using your detection functions
        # 3. Estimate & refine the rigid transformation.
        # Here we simulate some output.
        theta = 68.55
        t = (100.0, 200.0)
        inliers = 10
        st.write(f"Estimated Transformation: θ = {theta}, t = {t}")
        st.write(f"Number of inlier pairs: {inliers}")