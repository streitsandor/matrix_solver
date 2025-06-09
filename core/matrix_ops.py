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


def matrix_adjugate(saved_matrices) -> np.ndarray | None:
    """Adjungált mátrix számítása (kofaktor transzponáltja)."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print("Error: Matrix must be square.")
            return None
        n = A.shape[0]
        cofactors = np.zeros_like(A)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
                cofactors[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        return cofactors.T
    except Exception as e:
        print(f"Error during adjugate calculation: {e}")
        return None


def describe_matrix_types(saved_matrices) -> list[str] | None:
    """Felsorolja az adott mátrixra illő összes ismert típusát."""
    try:
        A = choose_matrix(saved_matrices)
        types = []

        if A.ndim != 2:
            print("Not a valid 2D matrix.")
            return None

        m, n = A.shape
        is_square = m == n

        if is_square:
            types.append("Négyzetes")

            if np.allclose(A, A.T):
                types.append("Szimmetrikus")

            if np.allclose(A.T @ A, np.eye(n)):
                types.append("Ortogonális")

            if np.allclose(A @ A, np.eye(n)):
                types.append("Involúciós")

            if np.allclose(A, np.eye(n)):
                types.append("Identitásmátrix")
                types.append("Diagonális")
                types.append("Szimmetrikus")
                types.append("Felső háromszög")
                types.append("Alsó háromszög")
                types.append("Skalár")

        if np.count_nonzero(A) == 0:
            types.append("Nullmátrix")

        if np.allclose(A, np.triu(A)):
            types.append("Felső háromszög")

        if np.allclose(A, np.tril(A)):
            types.append("Alsó háromszög")

        if np.count_nonzero(A - np.diag(np.diag(A))) == 0:
            types.append("Diagonális")
            if is_square and np.allclose(np.diag(A), A[0, 0] * np.ones(n)):
                types.append("Skalár")

        print("\nTípusok:")
        for t in sorted(set(types)):
            print(f"- {t}")
        return sorted(set(types))

    except Exception as e:
        print(f"Error during matrix type description: {e}")
        return None
