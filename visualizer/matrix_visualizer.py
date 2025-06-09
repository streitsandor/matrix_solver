import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core import get_matrix


def show_matrix_heatmap(matrix: np.ndarray, title: str = "Matrix Heatmap") -> None:
    """Heatmap vizualizálása."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()


def plot_matrix_sums(matrix: np.ndarray) -> None:
    """Sor és oszlop összegek vizualizálása."""
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].bar(range(len(row_sums)), row_sums, color="skyblue")
    axs[0].set_title("Row Sums")
    axs[0].set_xlabel("Row Index")
    axs[0].set_ylabel("Sum")

    axs[1].bar(range(len(col_sums)), col_sums, color="salmon")
    axs[1].set_title("Column Sums")
    axs[1].set_xlabel("Column Index")
    axs[1].set_ylabel("Sum")

    plt.tight_layout()
    plt.show()


def visualize_matrix(matrix: np.ndarray, name: str = "Matrix") -> None:
    """Vizualizációs felajánlása."""
    if matrix.size == 0:
        print("No matrix.")
        return

    visualize = input(f"Visualize the matrix '{name}'? (y/n): ").strip().lower()
    if visualize == "y":
        show_matrix_heatmap(matrix, title=f"{name} - Heatmap")
        plot_matrix_sums(matrix)


def visualize_matrix_option(saved_matrices: dict[str, np.ndarray], last_result) -> None:
    """Vizualizációs menü."""
    print("\nVisualize which matrix?")
    print("1. Saved matrix")
    print("2. Enter new matrix")
    print("3. Last result")
    choice = input("Choose option (1/2/3): ").strip()

    if choice == "1":
        name = input("Enter saved matrix name: ").strip()
        matrix = saved_matrices.get(name)
        if matrix is not None:
            visualize_matrix(matrix, name=f"Saved Matrix '{name}'")
        else:
            print("Matrix not found.")
    elif choice == "2":
        try:
            matrix = get_matrix("Enter matrix to visualize:")
            visualize_matrix(matrix, name="Typed Matrix")
        except Exception as e:
            print(f"Error: {e}")
    elif choice == "3":
        if last_result is not None:
            visualize_matrix(last_result, name="Last Result")
        else:
            print("No last result to visualize.")
    else:
        print("Invalid choice.")
