import numpy as np
from core import choose_matrix

forbidden_square = "Error: Matrix must be square."


def handle_binary_operation(op: str, saved_matrices) -> np.ndarray | None:
    """Bináris műveletek elvégzése (+, -, *, /) két mátrixon."""
    try:
        print("First matrix:")
        A = choose_matrix(saved_matrices)
        print("Second matrix:")
        B = choose_matrix(saved_matrices)

        if op in {"+", "-", "/"} and A.shape != B.shape:
            print("Error: Matrices must be the same shape.")
            return None

        if op == "+":
            return A + B
        elif op == "-":
            return A - B
        elif op == "*":
            A = np.atleast_2d(A)
            B = np.atleast_2d(B)
            if A.shape[1] != B.shape[0]:
                print(f"Error: Cannot multiply matrices of shapes {A.shape} and {B.shape}")
                return None
            return np.matmul(A, B)
        elif op == "/":
            return A / B
    except Exception as e:
        print(f"Error during operation: {e}")
        return None


def handle_unary_operation(op: str, saved_matrices) -> np.ndarray | None:
    """Egyéb műveletek elvégzése (square, sqrt) egy mátrixon."""
    try:
        A = choose_matrix(saved_matrices)
        if op == "square":
            if A.shape[0] != A.shape[1]:
                print(forbidden_square)
                return None
            return np.matmul(A, A)
        elif op == "sqrt":
            return np.sqrt(A)
    except Exception as e:
        print(f"Error during operation: {e}")
        return None


def scalar_multiply(saved_matrices) -> np.ndarray | None:
    """Mátrix skaláris szorzása."""
    try:
        scalar = float(input("Enter scalar value (number): ").strip())
        matrix = choose_matrix(saved_matrices)
        return scalar * matrix
    except ValueError:
        print("Invalid scalar.")
    except Exception as e:
        print(f"Error during scalar multiplication: {e}")
    return None


def matrix_inverse(saved_matrices) -> np.ndarray | None:
    """Mátrix inverzének számítása."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(forbidden_square)
            return None
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular and cannot be inverted.")
    except Exception as e:
        print(f"Error during inversion: {e}")
    return None


def matrix_determinant(saved_matrices) -> float | int | None:
    """Mátrix determinánsának számítása."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print("Error: Matrix must be square.")
            return None
        det = np.linalg.det(A)

        # Ha nagyon közel van egy integerhez, kerekítés
        if np.isclose(det, round(det)):
            return int(round(det))
        return det
    except Exception as e:
        print(f"Error during determinant calculation: {e}")
    return None
