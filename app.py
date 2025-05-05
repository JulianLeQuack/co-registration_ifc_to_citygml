import os, shutil, tempfile, streamlit as st
from pathlib import Path
import sys
import pickle

# define one temp folder
TMP_DIR = "./test_data/tmp"
# check if the tmp folder exists, if not create it
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

import matplotlib.pyplot as plt

# Import your footprint and registration functions.
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon, extract_building_storeys, extract_classes
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
                           "Point-symmetry Checking",
                           "Rigid Registration Estimation"])

if page == "File Upload":
    st.header("File Upload")
    col1, col2, col3 = st.columns(3)

    # --- CityGML upload ---
    with col1:
        st.subheader("CityGML File")
        citygml_fixed_path = os.path.join(TMP_DIR, "citygml_upload.gml")
        citygml_file = st.file_uploader("Upload CityGML file", type=["gml"], key="uploader_cgml")
        if citygml_file is not None:
            with open(citygml_fixed_path, "wb") as out:
                out.write(citygml_file.getvalue())
            st.session_state.citygml_path = citygml_fixed_path
            st.session_state.citygml_filename = citygml_file.name
            st.success(f"✅ CityGML saved:\n{citygml_fixed_path}")
        elif os.path.exists(citygml_fixed_path):
            st.session_state.citygml_path = citygml_fixed_path
            st.info(f"💾 Previously uploaded CityGML file found:\n{citygml_fixed_path}")
            if "citygml_filename" in st.session_state:
                st.info(f"File: {st.session_state.citygml_filename}")

    # --- IFC upload ---
    with col2:
        st.subheader("IFC File")
        ifc_fixed_path = os.path.join(TMP_DIR, "ifc_upload.ifc")
        ifc_file = st.file_uploader("Upload IFC file", type=["ifc"], key="uploader_ifc")
        if ifc_file is not None:
            with open(ifc_fixed_path, "wb") as out:
                out.write(ifc_file.getvalue())
            st.session_state.ifc_path = ifc_fixed_path
            st.session_state.ifc_filename = ifc_file.name
            st.success(f"✅ IFC saved:\n{ifc_fixed_path}")
        elif os.path.exists(ifc_fixed_path):
            st.session_state.ifc_path = ifc_fixed_path
            st.info(f"💾 Previously uploaded IFC file found:\n{ifc_fixed_path}")
            if "ifc_filename" in st.session_state:
                st.info(f"File: {st.session_state.ifc_filename}")

    # --- DXF upload ---
    with col3:
        st.subheader("DXF File")
        dxf_fixed_path = os.path.join(TMP_DIR, "dxf_upload.dxf")
        dxf_file = st.file_uploader("Upload DXF file", type=["dxf"], key="uploader_dxf")
        if dxf_file is not None:
            with open(dxf_fixed_path, "wb") as out:
                out.write(dxf_file.getvalue())
            st.session_state.dxf_path = dxf_fixed_path
            st.session_state.dxf_filename = dxf_file.name
            st.success(f"✅ DXF saved:\n{dxf_fixed_path}")
        elif os.path.exists(dxf_fixed_path):
            st.session_state.dxf_path = dxf_fixed_path
            st.info(f"💾 Previously uploaded DXF file found:\n{dxf_fixed_path}")
            if "dxf_filename" in st.session_state:
                st.info(f"File: {st.session_state.dxf_filename}")
        
# --- Footprint Creation ---
elif page == "Footprint Creation":
    st.header("Footprint Creation")
    col_cgml, col_ifc, col_dxf = st.columns(3)

    # --- CityGML column ---
    with col_cgml:
        st.subheader("CityGML Footprint")
        citygml_path = st.session_state.get("citygml_path")
        if not citygml_path:
            st.warning("Please upload a CityGML file on the File Upload page first.")
        else:
            # Define paths for persistent storage
            ids_all_path = os.path.join(TMP_DIR, "citygml_ids_all.pkl")
            sel_ids_path = os.path.join(TMP_DIR, "citygml_sel_ids.pkl")
            footprint_path = os.path.join(TMP_DIR, "citygml_footprint.pkl")

            # Load or extract building IDs
            if os.path.exists(ids_all_path):
                with open(ids_all_path, 'rb') as f:
                    building_ids_all = pickle.load(f)
            else:
                building_ids_all = extract_building_ids(citygml_path)
                with open(ids_all_path, 'wb') as f:
                    pickle.dump(building_ids_all, f)
            
            # Load previously selected IDs or initialize empty
            if os.path.exists(sel_ids_path):
                with open(sel_ids_path, 'rb') as f:
                    default_sel_ids = pickle.load(f)
            else:
                default_sel_ids = building_ids_all[0]

            # Building ID selection
            sel_ids = st.multiselect(
                "Select Building IDs",
                options=building_ids_all,
                default=default_sel_ids,
                key="citygml_sel_ids"
            )

            # Save selected IDs
            with open(sel_ids_path, 'wb') as f:
                pickle.dump(sel_ids, f)

            # Load existing footprint or create new one
            if len(sel_ids) > 0:
                with st.spinner("Rendering CityGML footprints…"):
                    if os.path.exists(footprint_path) and sel_ids == default_sel_ids:
                        # Load cached footprint if selection hasn't changed
                        with open(footprint_path, 'rb') as f:
                            mp = pickle.load(f)
                    else:
                        # Create new footprint if selection changed
                        mp = create_CityGML_footprint(
                            citygml_path=citygml_path,
                            building_ids=sel_ids,
                        )
                        # Cache the new footprint
                        with open(footprint_path, 'wb') as f:
                            pickle.dump(mp, f)
                    
                    st.session_state.citygml_footprint = mp

                    # Display footprint
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
        ifc_path = st.session_state.get("ifc_path")
        if not ifc_path:
            st.warning("Please upload an IFC file on the File Upload page first.")
        else:
            # Persistent storage paths for storeys, classes, and footprint
            storeys_all_path   = os.path.join(TMP_DIR, "ifc_storeys_all.pkl")
            sel_storeys_path   = os.path.join(TMP_DIR, "ifc_sel_storeys.pkl")
            ifc_classes_path   = os.path.join(TMP_DIR, "ifc_classes.pkl")
            ifc_sel_class_path = os.path.join(TMP_DIR, "ifc_sel_class.pkl")
            ifc_footprint_path = os.path.join(TMP_DIR, "ifc_footprint.pkl")

            # Load or extract all building storeys
            if os.path.exists(storeys_all_path):
                with open(storeys_all_path, "rb") as f:
                    ifc_storeys_all = pickle.load(f)
            else:
                ifc_storeys_all = extract_building_storeys(ifc_path)
                with open(storeys_all_path, "wb") as f:
                    pickle.dump(ifc_storeys_all, f)

            # Load previously selected storeys or default to first
            if os.path.exists(sel_storeys_path):
                with open(sel_storeys_path, "rb") as f:
                    default_storeys = pickle.load(f)
            else:
                default_storeys = []
            if not default_storeys and ifc_storeys_all:
                default_storeys = [ifc_storeys_all[0]]

            # Storey selection widget
            sel_storeys = st.multiselect(
                "Select IFC Building Storeys",
                options=ifc_storeys_all,
                default=default_storeys,
                key="ifc_storeys"
            )
            # Persist selection
            with open(sel_storeys_path, "wb") as f:
                pickle.dump(sel_storeys, f)

            # Load or extract all IFC classes
            if os.path.exists(ifc_classes_path):
                with open(ifc_classes_path, "rb") as f:
                    ifc_classes_all = pickle.load(f)
            else:
                ifc_classes_all = extract_classes(ifc_path)
                with open(ifc_classes_path, "wb") as f:
                    pickle.dump(ifc_classes_all, f)

            # Load previously selected class or default to "IfcSlab"
            if os.path.exists(ifc_sel_class_path):
                with open(ifc_sel_class_path, "rb") as f:
                    default_ifc_class = pickle.load(f)
            else:
                default_ifc_class = "IfcSlab"
            if default_ifc_class not in ifc_classes_all and ifc_classes_all:
                default_ifc_class = ifc_classes_all[0]

            # Class selection widget
            ifc_class = st.selectbox(
                "Select IFC class",
                options=ifc_classes_all,
                index=ifc_classes_all.index(default_ifc_class),
                key="ifc_type"
            )
            # Persist selection
            with open(ifc_sel_class_path, "wb") as f:
                pickle.dump(ifc_class, f)

            # Empty plot placeholder if nothing selected
            if not sel_storeys:
                st.info("Please select at least one storey above to render the IFC footprint.")
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                ax2.set_aspect("equal", "box")
                st.pyplot(fig2)
            else:
                # Load cached footprint or create new one
                with st.spinner("Rendering IFC footprint…"):
                    cache_valid = (
                        os.path.exists(ifc_footprint_path)
                        and sel_storeys == default_storeys
                        and ifc_class == default_ifc_class
                    )
                    if cache_valid:
                        with open(ifc_footprint_path, "rb") as f:
                            mp_ifc = pickle.load(f)
                    else:
                        mp_ifc = create_IFC_footprint_polygon(
                            ifc_path, ifc_class, building_storeys=sel_storeys
                        )
                        with open(ifc_footprint_path, "wb") as f:
                            pickle.dump(mp_ifc, f)
                    st.session_state.ifc_footprint = mp_ifc

                # Display IFC footprint
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                for poly in mp_ifc.geoms:
                    x, y = poly.exterior.xy
                    ax2.plot(x, y, color="green", linewidth=2)
                ax2.set_aspect("equal", "box")
                ax2.set_title(f"IFC Footprint ({ifc_class})")
                st.pyplot(fig2)

    # --- DXF column ---
    with col_dxf:
        st.subheader("DXF Footprint")
        dxf_path = st.session_state.get("dxf_path")
        if not dxf_path:
            st.warning("Please upload a DXF file on the File Upload page first.")
        else:
            # Persistent storage paths
            dxf_layers_path = os.path.join(TMP_DIR, "dxf_layers.pkl")
            dxf_sel_layer_path = os.path.join(TMP_DIR, "dxf_sel_layer.pkl")
            dxf_footprint_path = os.path.join(TMP_DIR, "dxf_footprint.pkl")

            # Load or extract all layers
            if os.path.exists(dxf_layers_path):
                with open(dxf_layers_path, "rb") as f:
                    dxf_layers_all = pickle.load(f)
            else:
                import ezdxf
                doc = ezdxf.readfile(dxf_path)
                dxf_layers_all = [ly.dxf.name for ly in doc.layers]
                with open(dxf_layers_path, "wb") as f:
                    pickle.dump(dxf_layers_all, f)

            # Load previously selected layer or default to first
            if os.path.exists(dxf_sel_layer_path):
                with open(dxf_sel_layer_path, "rb") as f:
                    default_dxf_layer = pickle.load(f)
            else:
                default_dxf_layer = "A_09_TRAGDECKE" if dxf_layers_all else ""

            # Layer selection
            dxf_layer = st.selectbox(
                "Select DXF layer",
                options=dxf_layers_all,
                index=dxf_layers_all.index(default_dxf_layer) if default_dxf_layer in dxf_layers_all else 0,
                key="dxf_sel_layer"
            )

            # Save selected layer
            with open(dxf_sel_layer_path, "wb") as f:
                pickle.dump(dxf_layer, f)

            # Load existing footprint or create new one
            with st.spinner("Rendering DXF footprints…"):
                if os.path.exists(dxf_footprint_path) and dxf_layer == default_dxf_layer:
                    with open(dxf_footprint_path, "rb") as f:
                        mp_dxf = pickle.load(f)
                else:
                    mp_dxf = cached_dxf(dxf_path, dxf_layer)
                    with open(dxf_footprint_path, "wb") as f:
                        pickle.dump(mp_dxf, f)
                st.session_state.dxf_footprint = mp_dxf

            fig3, ax3 = plt.subplots(figsize=(4, 4))
            for poly in mp_dxf.geoms:
                x, y = poly.exterior.xy
                ax3.plot(x, y, color="purple", linewidth=2)
            ax3.set_aspect("equal", "box")
            ax3.set_title(f"DXF Footprint ({dxf_layer})")
            st.pyplot(fig3)

# --- Corner Detection & Filtering ---
elif page == "Corner Detection & Filtering":
    st.header("Corner Detection & Filtering")
    col_cgml, col_ifc, col_dxf = st.columns(3)

    # --- CityGML column ---
    with col_cgml:
        st.subheader("CityGML")
        if "citygml_footprint" not in st.session_state:
            st.warning("No CityGML footprint available.")
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
            st.warning("No IFC footprint available.")
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
            st.warning("No DXF footprint available.")
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
                # 1) estimate
                rigid_transformation_ifc, inlier_pairs_ifc = estimate_rigid_transformation(
                    features_ifc_filtered, features_citygml_filtered,
                    distance_tol=distance_tol_ifc, angle_tol_deg=angle_tol_ifc
                )

                # 2) refine
                refined_ifc = refine_rigid_transformation(inlier_pairs_ifc)
                if refined_ifc is not None:
                    st.session_state.rigid_transformation_ifc = refined_ifc
                    st.success(f"Refined IFC → CityGML: θ={refined_ifc.theta:.3f}, t={refined_ifc.t}")
                else:
                    st.session_state.rigid_transformation_ifc = rigid_transformation_ifc
                    st.warning("Refinement failed, using initial estimate.")

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
                # 1) estimate
                rigid_transformation_dxf, inlier_pairs_dxf = estimate_rigid_transformation(
                    features_dxf_filtered, features_citygml_filtered,
                    distance_tol=distance_tol_dxf, angle_tol_deg=angle_tol_dxf
                )

                # 2) refine
                refined_dxf = refine_rigid_transformation(inlier_pairs_dxf)
                if refined_ifc is not None:
                    st.session_state.rigid_transformation_dxf = refined_dxf
                    st.success(f"Refined DXF → CityGML: θ={refined_dxf.theta:.3f}, t={refined_dxf.t}")
                else:
                    st.session_state.rigid_transformation_dxf = rigid_transformation_dxf
                    st.warning("Refinement failed, using initial estimate.")

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