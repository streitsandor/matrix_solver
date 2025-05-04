import numpy as np
from core import choose_matrix


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
                print("Error: Matrix must be square.")
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
