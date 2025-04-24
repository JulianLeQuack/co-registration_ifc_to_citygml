import streamlit as st
import matplotlib.pyplot as plt
import tempfile

# Import your footprint and registration functions.
from source.transformation_horizontal.create_footprints.create_IFC_footprint_polygon import create_IFC_footprint_polygon
from source.transformation_horizontal.create_footprints.create_DXF_footprint_polygon import create_DXF_footprint_polygon
from source.transformation_horizontal.create_footprints.create_CityGML_footprint import create_CityGML_footprint

from source.transformation_horizontal.detect_features import detect_features, filter_features_by_feature_triangle_area
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc") as tmp:
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
        ifc_type = st.selectbox("Select IFC Type", ["IfcSlab", "IfcWall"])
        if st.button("Create IFC Footprint"):
            with st.spinner("Generating footprint…"):
                footprint = create_IFC_footprint_polygon(
                    ifc_path=st.session_state.ifc_path,
                    ifc_type=ifc_type
                )
            if footprint:
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
    angle_threshold = st.slider("Corner Detection Angle Threshold (deg)", 10, 90, 30)
    filter_area = st.slider("Filter Triangle Area", 0.0, 50.0, 15.0)
    st.write("Current parameters:")
    st.write("Angle Threshold:", angle_threshold)
    st.write("Filter Triangle Area:", filter_area)
    
    # For demonstration, we simulate an already-created footprint and detected features.
    # Replace this dummy plot with a call to your actual functions.
    fig, ax = plt.subplots()
    # Dummy footprint: a rectangle
    x = [0, 1, 1, 0, 0]
    y = [0, 0, 1, 1, 0]
    ax.plot(x, y, color="blue", label="Footprint")
    # Dummy detected corners (red dots)
    ax.scatter([0, 1, 1, 0], [0, 0, 1, 1], color="red", label="Detected Corners")
    ax.set_title("Detected Features on Footprint")
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