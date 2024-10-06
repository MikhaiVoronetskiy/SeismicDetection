import main


def matrix_convolution(matrix, max_pooling=True, convolution_size=2, step=1):
    new_matrix = []
    amount_of_rows = len(matrix)
    amount_of_columns = len(matrix[0])
    for row_index in range(convolution_size-1, amount_of_rows, step):
        new_row = []
        for column_index in range(convolution_size-1, amount_of_columns, step):
            sum = 0
            for row in range(row_index, row_index - convolution_size, -1):
                for column in range(column_index, column_index - convolution_size, -1):
                    sum += matrix[row][column]
            if not max_pooling:
                sum = sum / convolution_size**2
            new_row.append(sum)
        new_matrix.append(new_row)
    return new_matrix


def matrix_to_vector(matrix):
    vector = []
    for row in matrix:
        for element in row:
            vector.append(element)
    return vector


