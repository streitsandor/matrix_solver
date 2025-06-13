import os
import numpy as np
from core import (
    saved_matrices,
    last_result,
    handle_binary_operation,
    handle_unary_operation,
    multiply_matrix_by_scalar,
    matrix_inverse,
    matrix_determinant,
    determinant_sarrus,
    determinant_laplace,
    matrix_adjugate,
    matrix_minor,
    matrix_cofactor,
    matrix_cofactor_matrix,
    matrix_rank,
    dodgson_condensation,
    solve_cramers_rule,
    solve_gaussian_elimination,
    solve_inverse_method,
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
    print("3. Matrix x Matrix Multiplication")
    print("4. Division")
    print("5. Power (Matrix ^ n)")
    print("6. Square Root (element-wise)")
    print("7. Matrix x Scalar Multiplication")
    print("8. Inverse of a matrix")
    print("9. Determinant of a matrix")
    print("10. Determinant (Sarrus' Rule)")
    print("11. Determinant (Laplace Expansion)")
    print("12. Adjugate of a matrix")
    print("13. Minor of a matrix")
    print("14. Cofactor of a matrix")
    print("15. Full Cofactor Matrix")
    print("16. Rank of a matrix")
    print("17. Dodgson Condensation")
    print("18. Linear solver (Cramer's Rule)")
    print("19. Linear solver (Gaussian Elimination)")
    print("20. Linear solver (Inverse Matrix Method)")
    print("21. Save matrix")
    print("22. Load matrix")
    print("23. Show saved matrices")
    print("24. Delete saved matrix")
    print("25. Describe matrix type")
    print("26. Visualize matrix")
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
        "5": lambda: handle_unary_operation("power", saved_matrices),
        "6": lambda: handle_unary_operation("sqrt", saved_matrices),
        "7": lambda: multiply_matrix_by_scalar(saved_matrices),
        "8": lambda: matrix_inverse(saved_matrices),
        "9": lambda: matrix_determinant(saved_matrices),
        "10": lambda: determinant_sarrus(saved_matrices),
        "11": lambda: determinant_laplace(saved_matrices),
        "12": lambda: matrix_adjugate(saved_matrices),
        "13": lambda: matrix_minor(saved_matrices),
        "14": lambda: matrix_cofactor(saved_matrices),
        "15": lambda: matrix_cofactor_matrix(saved_matrices),
        "16": lambda: matrix_rank(saved_matrices),
        "17": lambda: dodgson_condensation(saved_matrices),
        "18": lambda: solve_cramers_rule(saved_matrices),
        "19": lambda: solve_gaussian_elimination(saved_matrices),
        "20": lambda: solve_inverse_method(saved_matrices),
        "21": lambda: save_matrix(get_matrix, last_result),
        "22": load_matrix,
        "23": show_saved_matrices,
        "24": delete_saved_matrix,
        "25": lambda: describe_matrix_types(saved_matrices),
        "26": lambda: visualize_matrix_option(saved_matrices, last_result),
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
