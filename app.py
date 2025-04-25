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
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint, extract_building_ids

from source.transformation_horizontal.detect_features import detect_features, filter_features_by_triangle_area
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation


# Page setup
st.set_page_config(
        page_title="bim2city",
        page_icon="./images/bim2city_logo.png",
        layout="wide",
    )
st.title("bim2city")
# Sidebar navigation
page = st.sidebar.radio("Navigation", 
                          ["File Upload", 
                           "Footprint Creation", 
                           "Corner Detection & Filtering", 
                           "Rigid Registration Estimation"])

if page == "File Upload":
    st.header("File Upload")
    col1, col2, col3 = st.columns(3)

    # CityGML upload in first column
    with col1:
        st.subheader("CityGML File")
        citygml_file = st.file_uploader("Upload CityGML file", type=["gml"], key="uploader_cgml")
        if citygml_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gml", dir=str(TMP_DIR)) as tmp:
                tmp.write(citygml_file.getvalue())
                st.session_state.citygml_path = tmp.name
            st.success(f"✅ CityGML saved:\n{st.session_state.citygml_path}")

    # IFC upload in second column
    with col2:
        st.subheader("IFC File")
        ifc_file = st.file_uploader("Upload IFC file", type=["ifc"], key="uploader_ifc")
        if ifc_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc", dir=str(TMP_DIR)) as tmp:
                tmp.write(ifc_file.getvalue())
                st.session_state.ifc_path = tmp.name
            st.success(f"✅ IFC saved:\n{st.session_state.ifc_path}")

    # DXF upload in third column
    with col3:
        st.subheader("DXF File")
        dxf_file = st.file_uploader("Upload DXF file", type=["dxf"], key="uploader_dxf")
        if dxf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf", dir=str(TMP_DIR)) as tmp:
                tmp.write(dxf_file.getvalue())
                st.session_state.dxf_path = tmp.name
            st.success(f"✅ DXF saved:\n{st.session_state.dxf_path}")
        
elif page == "Footprint Creation":
    st.header("Footprint Creation")
    col_cgml, col_ifc, col_dxf = st.columns(3)

    # --- CityGML column ---
    with col_cgml:
        st.subheader("CityGML Footprint")
        if "citygml_path" not in st.session_state:
            st.warning("Please upload a CityGML file on the File Upload page first.")
        else:
            # 1) Extract all IDs once
            if "citygml_ids_all" not in st.session_state:
                st.session_state.citygml_ids_all = extract_building_ids(
                    st.session_state.citygml_path
                )
            # 2) Initialize selected IDs once
            if "citygml_sel_ids" not in st.session_state:
                st.session_state.citygml_sel_ids = []

            # 3) Multi-select widget
            sel_ids = st.multiselect(
                "Select Building IDs",
                options=st.session_state.citygml_ids_all,
                default=st.session_state.citygml_sel_ids,
                key="citygml_sel_ids"
            )

            # 4) Plot based on current selection
            with st.spinner("Rendering CityGML footprints…"):
                mp = create_CityGML_footprint(
                    citygml_path=st.session_state.citygml_path,
                    building_ids=sel_ids,
                )
            fig, ax = plt.subplots(figsize=(4, 4))
            for poly, bid in zip(mp.geoms, sel_ids):
                x, y = poly.exterior.xy
                ax.plot(x, y, color="blue", linewidth=2)
                cx, cy = poly.centroid.x, poly.centroid.y
                ax.text(cx, cy, bid, fontsize=8, ha="center", va="center")
            ax.set_aspect("equal", "box")
            ax.set_title("CityGML Footprint")
            st.pyplot(fig)

    # --- IFC column ---
    with col_ifc:
        st.subheader("IFC Footprint")
        if "ifc_path" not in st.session_state:
            st.warning("Please upload an IFC file on the File Upload page first.")
        else:
            if "ifc_type" not in st.session_state:
                st.session_state.ifc_type = "IfcSlab"

            st.selectbox(
                "Select IFC type",
                ["IfcSlab", "IfcWall"],
                index=["IfcSlab", "IfcWall"].index(st.session_state.ifc_type),
                key="ifc_type",
            )

            with st.spinner("Rendering IFC footprint…"):
                mp_ifc = create_IFC_footprint_polygon(
                    ifc_path=st.session_state.ifc_path,
                    ifc_type=st.session_state.ifc_type,
                )
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            for poly in mp_ifc.geoms:
                x, y = poly.exterior.xy
                ax2.plot(x, y, color="green", linewidth=2)
            ax2.set_aspect("equal", "box")
            ax2.set_title(f"IFC Footprint ({st.session_state.ifc_type})")
            st.pyplot(fig2)

    # --- DXF column (layers) ---
    with col_dxf:
        st.subheader("DXF Footprint")
        if "dxf_path" not in st.session_state:
            st.warning("Please upload a DXF file on the File Upload page first.")
        else:
            # extract layers once
            if "dxf_layers_all" not in st.session_state:
                import ezdxf
                doc = ezdxf.readfile(st.session_state.dxf_path)
                st.session_state.dxf_layers_all = [ly.name for ly in doc.layers]
            if "dxf_sel_layers" not in st.session_state:
                st.session_state.dxf_sel_layers = st.session_state.dxf_layers_all.copy()

            # plot first
            with st.spinner("Rendering DXF footprints…"):
                mp_dxf = create_DXF_footprint_polygon(
                    dxf_path=st.session_state.dxf_path,
                    layer_name=None  # the function can be adapted to accept multiple layers
                )
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            for poly in mp_dxf.geoms:
                x, y = poly.exterior.xy
                ax3.plot(x, y, color="purple", linewidth=2)
            ax3.set_aspect("equal", "box")
            ax3.set_title("DXF Footprint")
            st.pyplot(fig3)

            # multi‐select layers underneath
            st.session_state.dxf_sel_layers = st.multiselect(
                "Select DXF layers",
                options=st.session_state.dxf_layers_all,
                default=st.session_state.dxf_sel_layers,
                key="dxf_sel_layers",
            )

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