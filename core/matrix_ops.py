import numpy as np
from core import choose_matrix
from core.logger_setup import logger

FORBIDDEN_SQUARE = "Error: Matrix must be square."
ORIGINAL_MATRIX = "Original Matrix:\n%s"


def handle_binary_operation(op: str, saved_matrices) -> np.ndarray | None:
    """Bináris műveletek elvégzése (+, -, *, /) két mátrixon lépésenként."""
    try:
        print("First matrix:")
        A = choose_matrix(saved_matrices)

        print("Second matrix:")
        B = choose_matrix(saved_matrices)

        if op in {"+", "-", "/"} and A.shape != B.shape:
            error_msg = "Error: Matrices must be the same shape."
            print(error_msg)
            logger.error(error_msg)
            return None

        print("Matrix A:")
        print(A)
        logger.info("Matrix A:\n%s", A)

        print("Matrix B:")
        print(B)
        logger.info("Matrix B:\n%s", B)

        print(f"\nOperation: A {op} B")
        logger.info("Operation: A %s B", op)

        if op == "+":
            result = A + B
        elif op == "-":
            result = A - B
        elif op == "*":
            A = np.atleast_2d(A)
            B = np.atleast_2d(B)
            if A.shape[1] != B.shape[0]:
                error_msg = f"Error: Cannot multiply matrices of shapes {A.shape} and {B.shape}"
                print(error_msg)
                logger.error(error_msg)
                return None
            result = np.matmul(A, B)
        elif op == "/":
            result = A / B
        else:
            return None

        print("\nResult:")
        print(result)
        logger.info("Result:\n%s", result)
        return result
    except Exception as e:
        error_msg = f"Error during operation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def handle_unary_operation(op: str, saved_matrices) -> np.ndarray | None:
    """Egyéb műveletek elvégzése (square, sqrt) egy mátrixon lépésenként."""
    try:
        A = choose_matrix(saved_matrices)
        print(ORIGINAL_MATRIX)
        print(A)
        logger.info(ORIGINAL_MATRIX, A)

        if op == "square":
            if A.shape[0] != A.shape[1]:
                print(FORBIDDEN_SQUARE)
                logger.error(FORBIDDEN_SQUARE)
                return None
            result = np.matmul(A, A)
            print("\nStep: A squared = A * A")
            print(result)
            logger.info("Step: A squared = A * A\n%s", result)
            return result

        elif op == "sqrt":
            result = np.sqrt(A)
            print("\nStep: Element-wise square root")
            print(result)
            logger.info("Step: Element-wise square root\n%s", result)
            return result
    except Exception as e:
        error_msg = f"Error during operation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def scalar_multiply(saved_matrices) -> np.ndarray | None:
    """Mátrix skaláris szorzása lépésenként."""
    try:
        scalar = float(input("Enter scalar value (number): ").strip())
        matrix = choose_matrix(saved_matrices)
        print(ORIGINAL_MATRIX)
        print(matrix)
        logger.info(ORIGINAL_MATRIX, matrix)

        print(f"\nStep: Scalar multiplication with {scalar}")
        result = scalar * matrix
        print(result)
        logger.info("Scalar multiplication with %.2f\n%s", scalar, result)
        return result
    except ValueError:
        print("Invalid scalar.")
        logger.error("Invalid scalar input.")
    except Exception as e:
        error_msg = f"Error during scalar multiplication: {e}"
        print(error_msg)
        logger.error(error_msg)
    return None


def matrix_inverse(saved_matrices) -> np.ndarray | None:
    """Mátrix inverzének kiszámítása lépésenként."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX)
        print(A)
        logger.info(ORIGINAL_MATRIX, A)

        det = np.linalg.det(A)
        print(f"\nStep 1: Determinant = {det:.4f}")
        logger.info("Step 1: Determinant = %.4f", det)
        if np.isclose(det, 0):
            error_msg = "Error: Matrix is singular and cannot be inverted."
            print(error_msg)
            logger.error(error_msg)
            return None

        n = A.shape[0]
        cofactors = np.zeros_like(A)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
                cofactors[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
        adjugate = cofactors.T

        print("\nStep 2: Adjugate Matrix:")
        print(adjugate)
        logger.info("Step 2: Adjugate Matrix:\n%s", adjugate)

        inverse = (1 / det) * adjugate
        print("\nStep 3: Inverse Matrix:")
        print(inverse)
        logger.info("Step 3: Inverse Matrix:\n%s", inverse)

        return inverse
    except Exception as e:
        error_msg = f"Error during matrix inversion: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def matrix_determinant(saved_matrices) -> float | int | None:
    """Mátrix determinánsának számítása lépésenként."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX)
        print(A)
        logger.info(ORIGINAL_MATRIX, A)

        det = np.linalg.det(A)
        print(f"\nStep: Determinant = {det:.4f}")
        logger.info("Step: Determinant = %.4f", det)

        if np.isclose(det, round(det)):
            det = int(round(det))
            print(f"Rounded Determinant: {det}")
            logger.info("Rounded Determinant: %d", det)

        return det
    except Exception as e:
        error_msg = f"Error during determinant calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
    return None


def matrix_adjugate(saved_matrices) -> np.ndarray | None:
    """Adjungált mátrix számítása lépésenként."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX)
        print(A)
        logger.info(ORIGINAL_MATRIX, A)

        n = A.shape[0]
        cofactors = np.zeros_like(A)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
                cofactor = ((-1) ** (i + j)) * np.linalg.det(minor)
                cofactors[i, j] = cofactor
                print(f"Cofactor[{i}][{j}] = {cofactor:.4f}")
                logger.info("Cofactor[%d][%d] = %.4f", i, j, cofactor)

        adjugate = cofactors.T
        print("\nAdjugate Matrix:")
        print(adjugate)
        logger.info("Adjugate Matrix:\n%s", adjugate)
        return adjugate
    except Exception as e:
        error_msg = f"Error during adjugate calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def describe_matrix_types(saved_matrices) -> list[str] | None:
    """Felsorolja az adott mátrixra illő összes ismert típusát."""
    try:
        A = choose_matrix(saved_matrices)
        types = []

        print("\nAnalyzing Matrix:")
        print(A)
        logger.info("Analyzing Matrix:\n%s", A)

        if A.ndim != 2:
            error_msg = "Not a valid 2D matrix."
            print(error_msg)
            logger.error(error_msg)
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
                types.extend(
                    ["Identitásmátrix", "Diagonális", "Szimmetrikus", "Felső háromszög", "Alsó háromszög", "Skalár"]
                )

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
        logger.info("Detected Types: %s", sorted(set(types)))
        return sorted(set(types))

    except Exception as e:
        error_msg = f"Error during matrix type description: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None
