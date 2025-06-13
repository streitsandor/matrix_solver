import numpy as np
from core import choose_matrix
from core.logger_setup import logger

FORBIDDEN_SQUARE = "Error: Matrix must be square."
ORIGINAL_MATRIX_PRINT = "\nOriginal Matrix:"
ORIGINAL_MATRIX_LOG = "Original Matrix:\n%s"
COFACTOR_FORMAT = "Cofactor[%d][%d] = %.4f"


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
    """Egyéb műveletek elvégzése (power, sqrt) egy mátrixon lépésenként."""
    try:
        A = choose_matrix(saved_matrices)
        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        if op == "power":
            if A.shape[0] != A.shape[1]:
                print(FORBIDDEN_SQUARE)
                logger.error(FORBIDDEN_SQUARE)
                return None
            try:
                exp = int(input("Enter power (integer): ").strip())
            except ValueError:
                print("Invalid exponent.")
                logger.error("Invalid exponent input.")
                return None
            result = np.linalg.matrix_power(A, exp)
            print(f"\nStep: A^{exp} =")
            print(result)
            logger.info("Matrix raised to power %d:\n%s", exp, result)
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


def multiply_matrix_by_scalar(saved_matrices) -> np.ndarray | None:
    """Mátrix skaláris szorzása lépésenként."""
    try:
        scalar = float(input("Enter scalar value (number): ").strip())
        matrix = choose_matrix(saved_matrices)
        print(ORIGINAL_MATRIX_PRINT)
        print(matrix)
        logger.info(ORIGINAL_MATRIX_LOG, matrix)

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

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

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

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

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

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        n = A.shape[0]
        cofactors = np.zeros_like(A)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
                cofactor = ((-1) ** (i + j)) * np.linalg.det(minor)
                cofactors[i, j] = cofactor
                print(f"Cofactor[{i}][{j}] = {cofactor:.4f}")
                logger.info(COFACTOR_FORMAT, i, j, cofactor)

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

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

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
                    [
                        "Identitásmátrix",
                        "Diagonális",
                        "Szimmetrikus",
                        "Felső háromszög",
                        "Alsó háromszög",
                        "Skalár",
                    ]
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


def matrix_minor(saved_matrices) -> np.ndarray | None:
    """Kiszámítja egy elem minorját lépésenként a felhasználó által megadott sor és oszlop alapján (1-indexelve)."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        row = int(input(f"Enter row index to remove (1 to {A.shape[0]}): ").strip()) - 1
        col = int(input(f"Enter column index to remove (1 to {A.shape[1]}): ").strip()) - 1

        if not (0 <= row < A.shape[0] and 0 <= col < A.shape[1]):
            print("Invalid row or column index.")
            logger.error(
                "Invalid indices for minor calculation: row=%d, col=%d",
                row + 1,
                col + 1,
            )
            return None

        minor = np.delete(np.delete(A, row, axis=0), col, axis=1)
        det_minor = np.linalg.det(minor)

        print(f"\nMinor matrix by removing row {row + 1} and column {col + 1}:")
        print(minor)
        print(f"\nDeterminant of the minor matrix: {det_minor:.4f}")

        logger.info("Minor matrix (remove row %d, column %d):\n%s", row + 1, col + 1, minor)
        logger.info("Determinant of the minor matrix: %.4f", det_minor)

        return minor
    except Exception as e:
        error_msg = f"Error during minor calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def matrix_cofactor(saved_matrices) -> float | None:
    """Kiszámítja egy elem kofaktorát lépésenként a felhasználó által megadott sor és oszlop alapján (1-indexelve)."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print("\nOriginal Matrix:")
        print(A)
        logger.info("Original Matrix:\n%s", A)

        row = int(input(f"Enter row index (1 to {A.shape[0]}): ").strip()) - 1
        col = int(input(f"Enter column index (1 to {A.shape[1]}): ").strip()) - 1

        if not (0 <= row < A.shape[0] and 0 <= col < A.shape[1]):
            print("Invalid row or column index.")
            logger.error(
                "Invalid indices for cofactor calculation: row=%d, col=%d",
                row + 1,
                col + 1,
            )
            return None

        minor_matrix = np.delete(np.delete(A, row, axis=0), col, axis=1)
        minor_det = np.linalg.det(minor_matrix)
        cofactor = ((-1) ** (row + col)) * minor_det

        print(f"\nMinor matrix (remove row {row + 1}, column {col + 1}):")
        print(minor_matrix)
        print(f"Determinant of minor = {minor_det:.4f}")
        print(f"Cofactor[{row + 1}][{col + 1}] = (-1)^({row + 1}+{col + 1}) * Minor = {cofactor:.4f}")

        logger.info(
            "Minor matrix (remove row %d, column %d):\n%s",
            row + 1,
            col + 1,
            minor_matrix,
        )
        logger.info("Determinant of minor: %.4f", minor_det)
        logger.info(COFACTOR_FORMAT, row + 1, col + 1, cofactor)

        return cofactor
    except Exception as e:
        error_msg = f"Error during cofactor calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def matrix_cofactor_matrix(saved_matrices) -> np.ndarray | None:
    """Teljes kofaktormátrix kiszámítása lépésenként."""
    try:
        A = choose_matrix(saved_matrices)

        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        n = A.shape[0]
        cofactor_matrix = np.zeros_like(A)

        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
                cofactor = ((-1) ** (i + j)) * np.linalg.det(minor)
                cofactor_matrix[i, j] = cofactor
                print(f"Cofactor[{i+1}][{j+1}] = (-1)^({i+1}+{j+1}) * det(minor) = {cofactor:.4f}")
                logger.info(COFACTOR_FORMAT, i + 1, j + 1, cofactor)

        print("\nCofactor Matrix:")
        print(cofactor_matrix)
        logger.info("Cofactor Matrix:\n%s", cofactor_matrix)

        return cofactor_matrix
    except Exception as e:
        error_msg = f"Error during full cofactor matrix calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def matrix_rank(saved_matrices) -> int | None:
    """Mátrix rangjának kiszámítása lépésenként."""
    try:
        A = choose_matrix(saved_matrices)

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        rank = np.linalg.matrix_rank(A)

        print(f"\nStep: Matrix rank = {rank}")
        logger.info("Matrix rank = %d", rank)

        return rank
    except Exception as e:
        error_msg = f"Error during rank calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def dodgson_condensation(saved_matrices) -> float | None:
    """Dodgson-féle kondenzáció lépésenként, középső pivottal."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        n = A.shape[0]
        steps = [A.astype(float)]

        for k in range(1, n):
            prev = steps[-1]
            if k > 1:
                denom = steps[-2]
            else:
                denom = None

            size = prev.shape[0] - 1
            next_matrix = np.zeros((size, size))

            for i in range(size):
                for j in range(size):
                    a = prev[i, j]
                    b = prev[i, j + 1]
                    c = prev[i + 1, j]
                    d = prev[i + 1, j + 1]
                    numerator = a * d - b * c

                    if denom is None:
                        next_matrix[i, j] = numerator
                    else:
                        pivot = denom[i + 1, j + 1]
                        if np.isclose(pivot, 0):
                            print(f"Warning: Zero pivot at ({i+2},{j+2}) in step {k+1}, using fallback 1.0")
                            logger.warning("Zero pivot at (%d,%d) in step %d", i + 2, j + 2, k + 1)
                            pivot = 1.0
                        next_matrix[i, j] = numerator / pivot

            print(f"\nStep {k} matrix:")
            print(next_matrix)
            logger.info("Step %d matrix:\n%s", k, next_matrix)
            steps.append(next_matrix)

        determinant = steps[-1][0, 0]
        print(f"\nFinal determinant by Dodgson condensation: {determinant:.4f}")
        logger.info("Final determinant by Dodgson condensation: %.4f", determinant)
        return determinant

    except Exception as e:
        error_msg = f"Error during Dodgson condensation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def solve_cramers_rule(saved_matrices) -> np.ndarray | None:
    """Lineáris egyenletrendszer megoldása Cramer szabállyal."""
    try:
        print("Choose coefficient matrix A:")
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print("Choose constant matrix B (right-hand side):")
        B = choose_matrix(saved_matrices)

        if B.ndim == 1:
            B = B.reshape(-1, 1)

        if A.shape[0] != B.shape[0] or B.shape[1] != 1:
            print("B must be a column vector with the same number of rows as A.")
            logger.error("Invalid B shape: %s", B.shape)
            return None

        print(ORIGINAL_MATRIX_PRINT)
        print("A =")
        print(A)
        print("B =")
        print(B)
        logger.info("Cramer's Rule - Matrix A:\n%s", A)
        logger.info("Cramer's Rule - Matrix B:\n%s", B)

        det_A = np.linalg.det(A)
        if np.isclose(det_A, 0):
            print("Determinant of A is zero. System has no unique solution.")
            logger.error("Cramer's Rule failed: det(A) = 0")
            return None

        print(f"\nDeterminant of A: {det_A:.4f}")
        logger.info("Determinant of A: %.4f", det_A)

        n = A.shape[1]
        solution = np.zeros(n)

        for i in range(n):
            A_i = A.copy()
            A_i[:, i] = B[:, 0]
            det_A_i = np.linalg.det(A_i)

            print(f"\nA with column {i+1} replaced by B:")
            print(A_i)
            print(f"Determinant of A_{i+1}: {det_A_i:.4f}")
            logger.info("A with column %d replaced by B:\n%s", i + 1, A_i)
            logger.info("Determinant of A_%d: %.4f", i + 1, det_A_i)

            solution[i] = det_A_i / det_A

        print("\nSolution vector X:")
        print(solution.reshape(-1, 1))
        logger.info("Solution vector X:\n%s", solution.reshape(-1, 1))

        return solution.reshape(-1, 1)
    except Exception as e:
        error_msg = f"Error during Cramer's Rule calculation: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def round_if_close(value: float, tol: float = 0.05) -> float:
    """Tűréshatáron belüli kerekítés a legközelebbi egész számra."""
    return round(value) if abs(value - round(value)) < tol else round(value, 2)


def solve_gaussian_elimination(saved_matrices) -> np.ndarray | None:
    """Lineáris egyenletrendszer megoldása Gaussian eliminációval, 2 decimális kerekítéssel, x-y-z és jobb oldal + végső értékkel."""
    try:
        print("Choose coefficient matrix A:")
        A = choose_matrix(saved_matrices)

        print("Choose right-hand side matrix B:")
        B = choose_matrix(saved_matrices)

        A = A.astype(float)
        B = B.astype(float).reshape(-1, 1)

        if A.shape[0] != B.shape[0]:
            print("A and B must have the same number of rows.")
            logger.error("Mismatched rows: A%s, B%s", A.shape, B.shape)
            return None

        n = A.shape[0]
        augmented = np.hstack((A, B))

        print("\nOriginal Augmented Matrix [A | B]:")
        print(np.round(augmented, 2))
        logger.info("Original Augmented Matrix:\n%s", np.round(augmented, 2))

        # Forward elimination
        for i in range(n):
            max_row = i + np.argmax(abs(augmented[i:, i]))
            if i != max_row:
                augmented[[i, max_row]] = augmented[[max_row, i]]
                print(f"\nSwapped rows {i+1} and {max_row+1}:")
                print(np.round(augmented, 2))
                logger.info("Swapped rows %d and %d:\n%s", i + 1, max_row + 1, np.round(augmented, 2))

            pivot = augmented[i, i]
            if np.isclose(pivot, 0):
                print("Zero pivot encountered.")
                logger.error("Zero pivot at row %d", i + 1)
                return None
            augmented[i] /= pivot

            for j in range(i + 1, n):
                factor = augmented[j, i]
                augmented[j] -= factor * augmented[i]

        reduced = np.round(augmented, 2)

        print("\nFinal reduced matrix [A | B]:")
        print(reduced)
        logger.info("Final reduced matrix:\n%s", reduced)

        # Back-substitution
        x = np.zeros((n, 1))
        for i in range(n - 1, -1, -1):
            rhs = reduced[i, -1]
            coeffs = reduced[i, i + 1 : n].reshape(1, -1)
            known_x = x[i + 1 : n].reshape(-1, 1)
            x[i] = rhs - float(coeffs @ known_x)

        var_names = ["x", "y", "z", "w", "v", "u"]  # Extendable

        # Print variable values for each row
        print("\nEquation-wise results (with x, y, z and RHS):")
        for i in range(n):
            print(f"\nRow {i+1}:")
            for j in range(n):
                value = x[j] if j >= i else 0.0
                v = round_if_close(float(value))
                print(f" - {var_names[j]} = {v:.2f}")
            print(f" - right-hand side = {float(reduced[i, -1]):.2f}")

        print("\nFinal Solution:")
        for i in range(n):
            val = round_if_close(float(x[i, 0]))
            print(f" - {var_names[i]} = {val:.2f}")
        logger.info("Final solution vector:\n%s", x)

        return x

    except Exception as e:
        error_msg = f"Error during Gaussian elimination: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def solve_inverse_method(saved_matrices) -> np.ndarray | None:
    """Lineáris egyenletrendszer megoldása inverz módszerrel (A^-1 * B)."""
    try:
        print("Choose coefficient matrix A:")
        A = choose_matrix(saved_matrices)

        print("Choose right-hand side matrix B:")
        B = choose_matrix(saved_matrices)

        if A.shape[0] != A.shape[1]:
            print("A must be square.")
            logger.error("Non-square matrix A for inverse solution.")
            return None

        if A.shape[0] != B.shape[0]:
            print("A and B must have the same number of rows.")
            logger.error("Mismatched rows: A%s, B%s", A.shape, B.shape)
            return None

        det = np.linalg.det(A)
        if np.isclose(det, 0):
            print("Matrix A is singular.")
            logger.error("Singular matrix A for inverse solution.")
            return None

        A_inv = np.linalg.inv(A)
        X = A_inv @ B

        print("\nInverse of A:")
        print(A_inv)
        print("\nSolution vector X = A^-1 * B:")
        print(X)

        logger.info("Inverse of A:\n%s", A_inv)
        logger.info("Inverse method solution:\n%s", X)

        return X
    except Exception as e:
        error_msg = f"Error during inverse method solution: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def determinant_sarrus(saved_matrices) -> float | None:
    """Sarrus szabállyal determinánt számítás. (csak 3x3 mátrixra)."""
    try:
        A = choose_matrix(saved_matrices)
        if A.shape != (3, 3):
            print("Sarrus' Rule is only valid for 3x3 matrices.")
            logger.error("Invalid size for Sarrus' Rule: %s", A.shape)
            return None

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        a, b, c = A[0]
        d, e, f = A[1]
        g, h, i = A[2]

        pos1 = a * e * i
        pos2 = b * f * g
        pos3 = c * d * h
        pos_sum = pos1 + pos2 + pos3

        neg1 = c * e * g
        neg2 = a * f * h
        neg3 = b * d * i
        neg_sum = neg1 + neg2 + neg3

        print("\nPositive Diagonals:")
        print(f"  {a:.2f}·{e:.2f}·{i:.2f} + {b:.2f}·{f:.2f}·{g:.2f} + {c:.2f}·{d:.2f}·{h:.2f} = {pos_sum:.2f}")
        print("Negative Diagonals:")
        print(f"  {c:.2f}·{e:.2f}·{g:.2f} + {a:.2f}·{f:.2f}·{h:.2f} + {b:.2f}·{d:.2f}·{i:.2f} = {neg_sum:.2f}")

        determinant = pos_sum - neg_sum
        print(f"\nDeterminant (Sarrus): {determinant:.2f}")

        logger.info("Sarrus Rule: +%.2f - %.2f = %.2f", pos_sum, neg_sum, determinant)
        return round(determinant, 2)
    except Exception as e:
        error_msg = f"Error during Sarrus' Rule: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None


def determinant_laplace(saved_matrices) -> float | None:
    """Determináns számítás rekurzív Laplace módszerrel."""

    def laplace_recursive(matrix: np.ndarray) -> float:
        if matrix.shape == (2, 2):
            return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        total = 0
        for j in range(matrix.shape[1]):
            minor = np.delete(np.delete(matrix, 0, axis=0), j, axis=1)
            cofactor = ((-1) ** j) * matrix[0, j] * laplace_recursive(minor)
            print(f"cofactor for column {j+1}: {cofactor:.2f}")
            total += cofactor
        return total

    try:
        A = choose_matrix(saved_matrices)
        if A.shape[0] != A.shape[1]:
            print(FORBIDDEN_SQUARE)
            logger.error(FORBIDDEN_SQUARE)
            return None

        print(ORIGINAL_MATRIX_PRINT)
        print(A)
        logger.info(ORIGINAL_MATRIX_LOG, A)

        det = laplace_recursive(A)
        print(f"\nDeterminant (Laplace Expansion): {det:.2f}")
        logger.info("Laplace Expansion - determinant = %.2f", det)
        return round(det, 2)
    except Exception as e:
        error_msg = f"Error during Laplace expansion: {e}"
        print(error_msg)
        logger.error(error_msg)
        return None
