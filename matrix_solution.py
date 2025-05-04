import os
import numpy as np
from numpy._typing import NDArray
from typing import Union
from matrix_visualizer import visualize_matrix_option

saved_matrices: dict[str, NDArray] = {}
last_result: Union[NDArray, None] = None
missing_matrix = "Matrix not found."


def clear_console() -> None:
    """Console tisztítása."""
    os.system("cls" if os.name == "nt" else "clear")


def get_matrix(prompt: str = "Enter matrix (row by row, 'Q' to finish, 'T' to finish and transpose):") -> NDArray:
    """Mátrix létrehozása paraméterek bekérésével. Kilépés 'Q' vagy transzponáció 'T'."""
    print(prompt)
    matrix = []
    flag_transpose = False

    while True:
        row = input()
        keyword = row.strip().upper()
        if keyword == "Q":
            break
        elif keyword == "T":
            flag_transpose = True
            break
        try:
            matrix.append([float(x) for x in row.strip().split()])
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")

    if not matrix:
        raise ValueError("Empty matrix input.")

    result = np.array(matrix)
    return result.T if flag_transpose else result


def choose_matrix(prompt: str = "Use saved matrix? (y/n): ") -> NDArray:
    """Mátrix kiválasztása betöltéssel, vagy paraméterek bekérésével."""
    answer = input(prompt).strip().lower()
    if answer == "y":
        name = input("Enter matrix name to load: ").strip()
        if name in saved_matrices:
            return saved_matrices[name]
        else:
            print(missing_matrix)
            return choose_matrix(prompt)
    else:
        return get_matrix()


def menu() -> str:
    """Fő menü kirajzolása."""
    print("\nChoose a function:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Squaring (Matrix ^ 2)")
    print("6. Square Root (element-wise)")
    print("7. Multiply matrix with scalar")
    print("8. Save matrix")
    print("9. Load matrix")
    print("10. Show saved matrices")
    print("11. Delete saved matrix")
    print("12. Transpose saved or last matrix")
    print("13. Visualize matrix")
    print("Q. Quit")
    return input("Your choice: ").strip()


def handle_binary_operation(op: str) -> Union[NDArray, None]:
    """Bináris műveletek elvégzése (+, -, *, /) két mátrixon."""
    try:
        print("First matrix:")
        A = choose_matrix()
        print("Second matrix:")
        B = choose_matrix()

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


def handle_unary_operation(op: str) -> Union[NDArray, None]:
    """Egyéb műveletek elvégzése (square, sqrt) egy mátrixon."""
    try:
        A = choose_matrix()
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


def scalar_multiply() -> Union[NDArray, None]:
    """Mátrix skaláris szorzása."""
    try:
        scalar = input("Enter scalar value (number): ").strip()
        scalar = float(scalar)
        matrix = choose_matrix()
        return scalar * matrix
    except ValueError:
        print("Invalid scalar.")
    except Exception as e:
        print(f"Error during scalar multiplication: {e}")
    return None


def save_matrix() -> None:
    """Mátrix mentése memóriába."""
    global last_result
    try:
        use_result = input("Save the last result? (y/n): ").strip().lower()
        if use_result == "y":
            if last_result is None:
                print("Last result not found! Add it manually.")
                return
            mat = last_result
        else:
            mat = get_matrix("Enter new matrix to save:")

        name = input("Enter name to save this matrix: ").strip()
        if not name:
            print("Invalid name. Matrix not saved.")
            return

        if name in saved_matrices:
            print(f"Warning: Matrix '{name}' already exists. Overwriting.")
            print("Original:")
            print(saved_matrices[name])

        saved_matrices[name] = mat
        print(f"Matrix saved as '{name}'.")
    except Exception as e:
        print(f"Failed to save matrix: {e}")


def load_matrix() -> Union[NDArray, None]:
    """Név alapján mátrix visszaadása és kirajzolása."""
    name = input("Enter matrix name to load: ").strip()
    if name in saved_matrices:
        matrix = saved_matrices[name]
        print(f"\nMatrix '{name}':\n{matrix}")
        return matrix
    else:
        print(missing_matrix)
        return None


def show_saved_matrices() -> None:
    """Összes mentett mátrix kirajzolása."""
    if not saved_matrices:
        print("No matrices saved.")
    else:
        for name, mat in saved_matrices.items():
            print(f"\n{name}:\n{mat}")


def delete_saved_matrix() -> None:
    """Mentett mátrix törlése."""
    name = input("Enter matrix name to delete: ").strip()
    if name in saved_matrices:
        del saved_matrices[name]
        print(f"Matrix '{name}' deleted.")
    else:
        print(missing_matrix)


def transpose_matrix() -> None:
    """Utolsó, vagy mentett mátrix transzponálása."""
    global last_result
    source = input("Transpose (1) saved matrix or (2) last result? (1/2): ").strip()
    try:
        if source == "1":
            name = input("Enter saved matrix name: ").strip()
            if name not in saved_matrices:
                print(missing_matrix)
                return
            original = saved_matrices[name]
            transposed = original.T
            print(f"\nOriginal '{name}':\n{original}")
            print(f"\nTransposed:\n{transposed}")
            save = input("Save transposed matrix? (y/n): ").strip().lower()
            if save == "y":
                new_name = input("Enter new name for transposed matrix: ").strip()
                if new_name:
                    saved_matrices[new_name] = transposed
                    print(f"Transposed matrix saved as '{new_name}'.")
        elif source == "2":
            if last_result is None:
                print("No last result to transpose.")
                return
            transposed = last_result.T
            print("\nLast result:")
            print(last_result)
            print("\nTransposed:")
            print(transposed)
            save = input("Save transposed result? (y/n): ").strip().lower()
            if save == "y":
                name = input("Enter name to save transposed result: ").strip()
                if name:
                    saved_matrices[name] = transposed
                    print(f"Transposed result saved as '{name}'.")
        else:
            print("Invalid choice.")
    except Exception as e:
        print(f"Error during transpose: {e}")


def matrix_calculator() -> None:
    """Fő program logika. Függvények meghívása, menü navigálás, stb."""
    global last_result

    def quit_program() -> str:
        print("Goodbye!")
        return "quit"

    operations = {
        "1": lambda: handle_binary_operation("+"),
        "2": lambda: handle_binary_operation("-"),
        "3": lambda: handle_binary_operation("*"),
        "4": lambda: handle_binary_operation("/"),
        "5": lambda: handle_unary_operation("square"),
        "6": lambda: handle_unary_operation("sqrt"),
        "7": scalar_multiply,
        "8": save_matrix,
        "9": load_matrix,
        "10": show_saved_matrices,
        "11": delete_saved_matrix,
        "12": transpose_matrix,
        "13": lambda: visualize_matrix_option(saved_matrices, last_result),
        "Q": quit_program,
    }

    while True:
        choice = menu()
        result: Union[NDArray, None] = None

        action = operations.get(choice.upper())
        if action:
            if choice.upper() == "Q":
                action()
                break
            result = action() if callable(action) else None
        else:
            print("Invalid choice.")

        if result is not None:
            last_result = result
            print("\nResult:")
            print(result)


if __name__ == "__main__":
    """Program belépési pontja."""
    clear_console()
    matrix_calculator()
