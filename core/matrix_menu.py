import os
import numpy as np
from core import (
    saved_matrices,
    last_result,
    handle_binary_operation,
    handle_unary_operation,
    scalar_multiply,
    matrix_inverse,
    matrix_determinant,
    matrix_adjugate,
    matrix_minor,
    matrix_cofactor,
    save_matrix,
    get_matrix,
    load_matrix,
    show_saved_matrices,
    delete_saved_matrix,
    describe_matrix_types,
)
from visualizer import visualize_matrix_option


def clear_console() -> None:
    """Console tisztítása."""
    os.system("cls" if os.name == "nt" else "clear")


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
    print("8. Inverse of a matrix")
    print("9. Determinant of a matrix")
    print("10. Adjugate of a matrix")
    print("11. Minor of a matrix")
    print("12. Cofactor of a matrix")
    print("13. Save matrix")
    print("14. Load matrix")
    print("15. Show saved matrices")
    print("16. Delete saved matrix")
    print("17. Describe matrix type")
    print("18. Visualize matrix")
    print("Q. Quit")
    return input("Your choice: ").strip()


def matrix_calculator() -> None:
    """Fő program logika. Függvények meghívása, menü navigálás, stb."""
    global last_result

    clear_console()

    def quit_program() -> str:
        print("Goodbye!")
        return "quit"

    operations = {
        "1": lambda: handle_binary_operation("+", saved_matrices),
        "2": lambda: handle_binary_operation("-", saved_matrices),
        "3": lambda: handle_binary_operation("*", saved_matrices),
        "4": lambda: handle_binary_operation("/", saved_matrices),
        "5": lambda: handle_unary_operation("square", saved_matrices),
        "6": lambda: handle_unary_operation("sqrt", saved_matrices),
        "7": lambda: scalar_multiply(saved_matrices),
        "8": lambda: matrix_inverse(saved_matrices),
        "9": lambda: matrix_determinant(saved_matrices),
        "10": lambda: matrix_adjugate(saved_matrices),
        "11": lambda: matrix_minor(saved_matrices),
        "12": lambda: matrix_cofactor(saved_matrices),
        "13": lambda: save_matrix(get_matrix, last_result),
        "14": load_matrix,
        "15": show_saved_matrices,
        "16": delete_saved_matrix,
        "17": lambda: describe_matrix_types(saved_matrices),
        "18": lambda: visualize_matrix_option(saved_matrices, last_result),
        "Q": quit_program,
    }

    while True:
        choice = menu()
        result = None
        action = operations.get(choice.upper())
        if action:
            if choice.upper() == "Q":
                action()
                break
            result = action()
        else:
            print("Invalid choice.")

        if isinstance(result, np.ndarray):
            print("\nResult:")
            print(result)
            last_result = result
        elif isinstance(result, (float, int, np.float64, np.int64)):  # type: ignore
            print(f"\nResult: {result}")
            last_result = result
