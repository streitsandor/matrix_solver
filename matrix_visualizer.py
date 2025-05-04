import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def show_matrix_heatmap(matrix: np.ndarray, title: str = "Matrix Heatmap") -> None:
    """Displays a heatmap of the given matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()


def plot_matrix_sums(matrix: np.ndarray) -> None:
    """Displays bar charts of row and column sums."""
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
    """Optional interface to visualize a matrix with both heatmap and sums."""
    if matrix.size == 0:
        print("No matrix.")
        return

    visualize = input(f"Would you like to visualize the matrix '{name}'? (y/n): ").strip().lower()
    if visualize == "y":
        show_matrix_heatmap(matrix, title=f"{name} - Heatmap")
        plot_matrix_sums(matrix)
