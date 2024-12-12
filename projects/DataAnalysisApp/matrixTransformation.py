import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def apply_transformation(points, matrix):
    """Apply transformation matrix to a set of points."""
    return np.dot(points, matrix.T)

def plot_shape(original_points, transformed_points):
    """Plot original and transformed shapes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original shape
    ax1.plot(original_points[:, 0], original_points[:, 1], 'b-')
    ax1.fill(original_points[:, 0], original_points[:, 1], alpha=0.3)
    ax1.grid(True)
    ax1.set_title('Original Shape')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Plot transformed shape
    ax2.plot(transformed_points[:, 0], transformed_points[:, 1], 'r-')
    ax2.fill(transformed_points[:, 0], transformed_points[:, 1], alpha=0.3)
    ax2.grid(True)
    ax2.set_title('Transformed Shape')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    return fig

def main():
    st.title("Matrix Transformation Visualizer")
    st.write("Adjust the transformation matrix values to see how they affect the shape.")

    # Create a square as the default shape
    square = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])

    # Matrix input
    st.sidebar.header("Transformation Matrix")
    a = st.sidebar.slider("a (1,1)", -2.0, 2.0, 1.0, 0.1)
    b = st.sidebar.slider("b (1,2)", -2.0, 2.0, 0.0, 0.1)
    c = st.sidebar.slider("c (2,1)", -2.0, 2.0, 0.0, 0.1)
    d = st.sidebar.slider("d (2,2)", -2.0, 2.0, 1.0, 0.1)

    # Create transformation matrix
    transformation_matrix = np.array([[a, b], [c, d]])

    # Display the matrix
    st.write("Current Transformation Matrix:")
    st.write(transformation_matrix)

    # Apply transformation
    transformed_square = apply_transformation(square, transformation_matrix)

    # Plot the shapes
    fig = plot_shape(square, transformed_square)
    st.pyplot(fig)

    # Add explanation
    st.markdown("""
    ### How it works:
    - The left plot shows the original square
    - The right plot shows the transformed shape
    - Adjust the matrix values in the sidebar to see different transformations:
        - Scale: Change diagonal values (a, d)
        - Rotation: Set opposite values for b and c
        - Shear: Modify b or c while keeping others at default
    """)

if __name__ == "__main__":
    main()