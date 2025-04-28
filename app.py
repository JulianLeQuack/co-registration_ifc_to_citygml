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

from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area
from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation, refine_rigid_transformation

@st.cache_data(show_spinner=False)
def cached_ifc(path, ifc_type):
    return create_IFC_footprint_polygon(path, ifc_type)

@st.cache_data(show_spinner=False)
def cached_dxf(path, layer):
    return create_DXF_footprint_polygon(path, layer)

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
        
# --- Footprint Creation ---
elif page == "Footprint Creation":
    st.header("Footprint Creation")
    col_cgml, col_ifc, col_dxf = st.columns(3)

    # --- CityGML column ---
    with col_cgml:
        st.subheader("CityGML Footprint")
        if "citygml_path" not in st.session_state:
            st.warning("Please upload a CityGML file on the File Upload page first.")
        else:
            if "citygml_ids_all" not in st.session_state:
                st.session_state.citygml_ids_all = extract_building_ids(st.session_state.citygml_path)
            if "citygml_sel_ids" not in st.session_state:
                st.session_state.citygml_sel_ids = []
            sel_ids = st.multiselect(
                "Select Building IDs",
                options=st.session_state.citygml_ids_all,
                default=st.session_state.citygml_sel_ids,
                key="citygml_sel_ids"
            )
            with st.spinner("Rendering CityGML footprints…"):
                mp = create_CityGML_footprint(
                    citygml_path=st.session_state.citygml_path,
                    building_ids=sel_ids,
                )
                st.session_state.citygml_footprint = mp
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
                mp_ifc = cached_ifc(st.session_state.ifc_path, st.session_state.ifc_type)
                st.session_state.ifc_footprint = mp_ifc
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            for poly in mp_ifc.geoms:
                x, y = poly.exterior.xy
                ax2.plot(x, y, color="green", linewidth=2)
            ax2.set_aspect("equal", "box")
            ax2.set_title(f"IFC Footprint ({st.session_state.ifc_type})")
            st.pyplot(fig2)

    # --- DXF column ---
    with col_dxf:
        st.subheader("DXF Footprint")
        if "dxf_path" not in st.session_state:
            st.warning("Please upload a DXF file on the File Upload page first.")
        else:
            if "dxf_layers_all" not in st.session_state:
                import ezdxf
                doc = ezdxf.readfile(st.session_state.dxf_path)
                st.session_state.dxf_layers_all = [ly.dxf.name for ly in doc.layers]
            if "dxf_sel_layer" not in st.session_state:
                st.session_state.dxf_sel_layer = st.session_state.dxf_layers_all[0]
            st.selectbox(
                "Select DXF layer",
                options=st.session_state.dxf_layers_all,
                index=st.session_state.dxf_layers_all.index(st.session_state.dxf_sel_layer),
                key="dxf_sel_layer",
            )
            with st.spinner("Rendering DXF footprints…"):
                mp_dxf = cached_dxf(st.session_state.dxf_path, st.session_state.dxf_sel_layer)
                st.session_state.dxf_footprint = mp_dxf
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            for poly in mp_dxf.geoms:
                x, y = poly.exterior.xy
                ax3.plot(x, y, color="purple", linewidth=2)
            ax3.set_aspect("equal", "box")
            ax3.set_title(f"DXF Footprint ({st.session_state.dxf_sel_layer})")
            st.pyplot(fig3)

# --- Corner Detection & Filtering ---
elif page == "Corner Detection & Filtering":
    st.header("Corner Detection & Filtering")
    col_cgml, col_ifc, col_dxf = st.columns(3)

    # --- CityGML column ---
    with col_cgml:
        st.subheader("CityGML")
        if "citygml_footprint" not in st.session_state:
            st.info("No CityGML footprint available.")
        else:
            angle_threshold = st.slider(
                "Corner Detection Angle Threshold (deg)", 0, 180, 30, key="angle_threshold_cgml"
            )
            min_area = st.slider(
                "Filter Triangle Area", 0.0, 100.0, 15.0, key="min_area_cgml"
            )
            with st.spinner("Detecting and filtering features…"):
                detected = detect_features(st.session_state.citygml_footprint, angle_threshold_deg=angle_threshold)
                filtered = filter_features_by_feature_triangle_area(detected, min_area)
                st.session_state.citygml_features_filtered = filtered
            fig, ax = plt.subplots(figsize=(4, 4))
            for poly in st.session_state.citygml_footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="blue", lw=1)
            if detected.size:
                ax.scatter(detected[:,2], detected[:,3], color="orange", label=f"Detected ({len(detected)})")
            if filtered.size:
                ax.scatter(filtered[:,2], filtered[:,3], color="red", label=f"Filtered ({len(filtered)})")
            ax.set_title(f"CityGML: angle≥{angle_threshold}° & area≥{min_area}")
            ax.set_aspect("equal", "box")
            ax.legend()
            st.pyplot(fig)

    # --- IFC column ---
    with col_ifc:
        st.subheader("IFC")
        if "ifc_footprint" not in st.session_state:
            st.info("No IFC footprint available.")
        else:
            angle_threshold = st.slider(
                "Corner Detection Angle Threshold (deg)", 0, 180, 30, key="angle_threshold_ifc"
            )
            min_area = st.slider(
                "Filter Triangle Area", 0.0, 100.0, 15.0, key="min_area_ifc"
            )
            with st.spinner("Detecting and filtering features…"):
                detected = detect_features(st.session_state.ifc_footprint, angle_threshold_deg=angle_threshold)
                filtered = filter_features_by_feature_triangle_area(detected, min_area)
                st.session_state.ifc_features_filtered = filtered
            fig, ax = plt.subplots(figsize=(4, 4))
            for poly in st.session_state.ifc_footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="green", lw=1)
            if detected.size:
                ax.scatter(detected[:,2], detected[:,3], color="orange", label=f"Detected ({len(detected)})")
            if filtered.size:
                ax.scatter(filtered[:,2], filtered[:,3], color="red", label=f"Filtered ({len(filtered)})")
            ax.set_title(f"IFC: angle≥{angle_threshold}° & area≥{min_area}")
            ax.set_aspect("equal", "box")
            ax.legend()
            st.pyplot(fig)

    # --- DXF column ---
    with col_dxf:
        st.subheader("DXF")
        if "dxf_footprint" not in st.session_state:
            st.info("No DXF footprint available.")
        else:
            angle_threshold = st.slider(
                "Corner Detection Angle Threshold (deg)", 0, 180, 30, key="angle_threshold_dxf"
            )
            min_area = st.slider(
                "Filter Triangle Area", 0.0, 100.0, 15.0, key="min_area_dxf"
            )
            with st.spinner("Detecting and filtering features…"):
                detected = detect_features(st.session_state.dxf_footprint, angle_threshold_deg=angle_threshold)
                filtered = filter_features_by_feature_triangle_area(detected, min_area)
                st.session_state.dxf_features_filtered = filtered
            fig, ax = plt.subplots(figsize=(4, 4))
            for poly in st.session_state.dxf_footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="purple", lw=1)
            if detected.size:
                ax.scatter(detected[:,2], detected[:,3], color="orange", label=f"Detected ({len(detected)})")
            if filtered.size:
                ax.scatter(filtered[:,2], filtered[:,3], color="red", label=f"Filtered ({len(filtered)})")
            ax.set_title(f"DXF: angle≥{angle_threshold}° & area≥{min_area}")
            ax.set_aspect("equal", "box")
            ax.legend()
            st.pyplot(fig)

# --- Rigid Registration Estimation ---
elif page == "Rigid Registration Estimation":
    st.header("Rigid Registration Estimation")
    col_ifc, col_dxf = st.columns(2)

    # --- IFC to CityGML Registration ---
    with col_ifc:
        st.subheader("IFC → CityGML")
        distance_tol_ifc = st.number_input("Distance tolerance (IFC)", min_value=0.0, value=1.0, step=0.1, key="dist_tol_ifc")
        angle_tol_ifc = st.number_input("Angle tolerance (deg, IFC)", min_value=0, value=45, step=1, key="angle_tol_ifc")

        # Print transformation if available
        if "rigid_transformation_ifc" in st.session_state and st.session_state.rigid_transformation_ifc is not None:
            rt = st.session_state.rigid_transformation_ifc
            st.markdown(rt)

        if st.button("Run IFC → CityGML Registration", key="run_ifc_reg"):
            features_ifc_filtered = st.session_state.get("ifc_features_filtered", None)
            features_citygml_filtered = st.session_state.get("citygml_features_filtered", None)
            if features_ifc_filtered is None or features_citygml_filtered is None:
                st.error("Please extract and filter features on the previous page first.")
            else:
                from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation
                rigid_transformation_ifc, inlier_pairs_ifc = estimate_rigid_transformation(
                    features_ifc_filtered, features_citygml_filtered,
                    distance_tol=distance_tol_ifc, angle_tol_deg=angle_tol_ifc
                )
                st.session_state.rigid_transformation_ifc = rigid_transformation_ifc

        # Only plot IFC→CityGML in this column
        if "rigid_transformation_ifc" in st.session_state and st.session_state.rigid_transformation_ifc is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            for poly in st.session_state.citygml_footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="blue", label="CityGML" if 'CityGML' not in ax.get_legend_handles_labels()[1] else "")
            transformed = st.session_state.rigid_transformation_ifc.transform(st.session_state.ifc_footprint)
            for poly in transformed.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="green", linestyle="--", label="IFC (transformed)" if 'IFC (transformed)' not in ax.get_legend_handles_labels()[1] else "")
            ax.set_title("IFC → CityGML Registration")
            ax.set_aspect("equal", "box")
            ax.legend()
            st.pyplot(fig)

    # --- DXF to CityGML Registration ---
    with col_dxf:
        st.subheader("DXF → CityGML")
        distance_tol_dxf = st.number_input("Distance tolerance (DXF)", min_value=0.0, value=1.0, step=0.1, key="dist_tol_dxf")
        angle_tol_dxf = st.number_input("Angle tolerance (deg, DXF)", min_value=0, value=45, step=1, key="angle_tol_dxf")

        # Print transformation if available
        if "rigid_transformation_dxf" in st.session_state and st.session_state.rigid_transformation_dxf is not None:
            rt = st.session_state.rigid_transformation_dxf
            st.markdown(rt)

        if st.button("Run DXF → CityGML Registration", key="run_dxf_reg"):
            features_dxf_filtered = st.session_state.get("dxf_features_filtered", None)
            features_citygml_filtered = st.session_state.get("citygml_features_filtered", None)
            if features_dxf_filtered is None or features_citygml_filtered is None:
                st.error("Please extract and filter features on the previous page first.")
            else:
                from source.transformation_horizontal.estimate_rigid_transformation import estimate_rigid_transformation
                rigid_transformation_dxf, inlier_pairs_dxf = estimate_rigid_transformation(
                    features_dxf_filtered, features_citygml_filtered,
                    distance_tol=distance_tol_dxf, angle_tol_deg=angle_tol_dxf
                )
                st.session_state.rigid_transformation_dxf = rigid_transformation_dxf

        # Only plot DXF→CityGML in this column
        if "rigid_transformation_dxf" in st.session_state and st.session_state.rigid_transformation_dxf is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            for poly in st.session_state.citygml_footprint.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="blue", label="CityGML" if 'CityGML' not in ax.get_legend_handles_labels()[1] else "")
            transformed = st.session_state.rigid_transformation_dxf.transform(st.session_state.dxf_footprint)
            for poly in transformed.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="purple", linestyle="--", label="DXF (transformed)" if 'DXF (transformed)' not in ax.get_legend_handles_labels()[1] else "")
            ax.set_title("DXF → CityGML Registration")
            ax.set_aspect("equal", "box")
            ax.legend()
            st.pyplot(fig)