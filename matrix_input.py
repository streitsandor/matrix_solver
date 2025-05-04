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
