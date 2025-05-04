import numpy as np

saved_matrices: dict[str, np.ndarray] = {}
last_result = None


def save_matrix(get_matrix, last_result_ref) -> None:
    """Mátrix mentése memóriába."""
    try:
        use_result = input("Save the last result? (y/n): ").strip().lower()
        mat = last_result_ref if use_result == "y" else get_matrix("Enter new matrix to save:")

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


def load_matrix() -> np.ndarray | None:
    """Név alapján mátrix visszaadása és kirajzolása."""
    name = input("Enter matrix name to load: ").strip()
    if name in saved_matrices:
        matrix = saved_matrices[name]
        print(f"\nMatrix '{name}':\n{matrix}")
        return matrix
    else:
        print("Matrix not found.")
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
        print("Matrix not found.")
