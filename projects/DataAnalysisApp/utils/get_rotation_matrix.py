"""
Computes a rotation matrix for the given input matrix and rotates it by the specified angles around the X, Y, and Z axes.
Args:
    input_matrix (list or np.ndarray): The input matrix to be rotated. It should be a 1D array that can be reshaped into a matrix with `row_count` rows.
    row_count (int): The number of rows in the input matrix. Must be between 1 and 3.
    rotationX (float): The rotation angle around the X-axis in degrees.
    rotationY (float): The rotation angle around the Y-axis in degrees.
    rotationZ (float): The rotation angle around the Z-axis in degrees.
Returns:
    np.ndarray: The rotated matrix, flattened back to a 1D array.
Raises:
    ValueError: If `row_count` is less than 1 or greater than 3.
    ValueError: If `input_matrix` is None or does not contain at least 1 item per row.
This function takes an input matrix and rotates it by the specified angles around the X, Y, and Z axes.
"""
import numpy as np 


def get_rotation_matrix(input_matrix, row_count, rotationX, rotationY, rotationZ):
        if row_count < 1 or row_count > 3:
            raise ValueError("row_count must be greater than zero and less than four.")
        if input_matrix is None or len(input_matrix) < row_count:
            raise ValueError("input_matrix must contain at least 1 item per row.")

        # Convert angles from degrees to radians
        rotationX = np.radians(rotationX)
        rotationY = np.radians(rotationY)
        rotationZ = np.radians(rotationZ)

        # Compute trigonometric values for the rotation matrix
        cosA, sinA = np.cos(rotationX), np.sin(rotationX)
        cosB, sinB = np.cos(rotationY), np.sin(rotationY)
        cosC, sinC = np.cos(rotationZ), np.sin(rotationZ)

        # Build the 3x3 rotation matrix
        rotation_matrix = np.array([
            [cosA * cosB * cosC - sinA * sinC, -cosA * cosB * sinC - sinA * cosC, cosA * sinB],
            [sinA * cosB * cosC + cosA * sinC, -sinA * cosB * sinC + cosA * cosC, sinA * sinB],
            [-sinB * cosC, sinB * sinC, cosB]
        ]).squeeze()

        # Reshape the input to matrix form with row_count rows
        input_matrix = np.array(input_matrix).reshape((row_count, -1))

        # Perform matrix multiplication
        result_matrix = rotation_matrix @ input_matrix

        # Flatten the result back to a 1D array to match the original output format
        return result_matrix.flatten()