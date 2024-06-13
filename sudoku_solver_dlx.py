import numpy as np
import time


class Node():
    # class definition for dancing link nodes
    def __init__(self, row = -1, col = -1):
        self.right = self.left = self.up = self.down = self
        self.column = self
        self.row = row 
        self.col = col 

class columnHeader(Node):
    # derived class for column header nodes of dancing links
    def __init__(self, col):
        super().__init__(col = col)
        self.size = 0      # No. of 1s in the column


class dancing_links():
    # class definition for creating dancing links structure
    def __init__(self, binary_matrix):
        self.header = columnHeader(-1)
        self.columns = []

        # creating circular doubly linked list between all column headers 
        for col in range(len(binary_matrix[0])):
            new_column = columnHeader(col)
            self.columns.append(new_column)
            self.header.left.right = new_column
            new_column.left = self.header.left
            new_column.right = self.header
            self.header.left = new_column

        # creating circular doubly linked list between all the nodes in the dancing link
        for row in range (len(binary_matrix)):
            prev_node = None
            row_header = None
            for col in range(len(binary_matrix[0])):
                if (binary_matrix[row][col] == 1):
                    new_node = Node(row, col)
                    column_header = self.columns[col]

                    # Vertical link
                    new_node.down = column_header
                    new_node.up = column_header.up
                    column_header.up.down = new_node
                    column_header.up = new_node

                    # Horizontal link
                    if prev_node is not None:
                        prev_node.right = new_node
                        new_node.left = prev_node
                    else:
                        row_header = new_node
                    prev_node = new_node

                    # Link to column header
                    new_node.column = column_header
                    column_header.size += 1

            if prev_node is not None:
                prev_node.right = row_header
                row_header.left = prev_node

    # function to cover any specified column of dancing links 
    def cover_column(self, col_node):
        col_node.right.left = col_node.left
        col_node.left.right = col_node.right
        node = col_node.down
        while (node != col_node):
            right_node = node.right
            while (right_node != node):
                right_col = self.columns[right_node.col]
                right_node.down.up = right_node.up
                right_node.up.down = right_node.down
                right_col.size = right_col.size - 1
                right_node = right_node.right
            node = node.down

    # function to uncover any specified column of dancing link
    def uncover_column(self, col_node):
        node = col_node.up
        while (node != col_node):
            left_node = node.left
            while (left_node != node):
                left_col = self.columns[left_node.col]
                left_col.size = left_col.size + 1
                left_node.down.up = left_node
                left_node.up.down = left_node
                left_node = left_node.left
            node = node.up

        col_node.right.left = col_node
        col_node.left.right = col_node
   
# returns the row of binary matrix with respective 0's and 1's based on the passed row no., column no. and value in the cell of given sudoku puzzle 
def cover_cell(N, n_row, n_col, num):
    row = []
    for i in range(4*N*N):
        if ((i == n_row*N + n_col) or (i == N*N + n_row*N + num - 1) or (i == 2*N*N + n_col*N + num - 1) or (i == 3*N*N + (n_row//3 * 3 + n_col//3)*N + num - 1)):
            row.append(1)
        else:
            row.append(0)
    return row

# function to create a binary matrix from the given sudoku grid to represent an exact cover problem
def create_binary_matrix(board):
    N = len(board)
    binary_matrix = []
    for i in range(N):
        for j in range(N):
            if (board[i][j] != 0):
                for num in range(1, N+1):
                    # places 1 in the corresponding cell for the given cell value and places a zero row vector on every other value
                    if (num == board[i][j]):
                        binary_matrix.append(cover_cell(N, i, j, board[i][j]))
                    else:
                        binary_matrix.append([0]*(4*N*N))
            else:
                for num in range(1, N+1): # places 1 in the corresponding cell of the binary matrix for each of 9 possible values
                    binary_matrix.append(cover_cell(N, i, j, num))

    return binary_matrix

# solves the sudoku using the dancing link algorithm (Algorithm-X)
def exact_cover_solution(solution, dl):
    if (dl.header.right == dl.header):
        print("Solution found")
        return True
    else:
        min_size = float('inf')
        col_node = dl.header.right
        chosen_col = col_node
        # chooses column with minimum no. of 1 
        while (col_node != dl.header):
            if (col_node.size < min_size):
                min_size = col_node.size
                chosen_col = col_node
            col_node = col_node.right
        if min_size == 0:
            return False
        dl.cover_column(chosen_col)

        node = chosen_col.down
        while(node != chosen_col):
            solution.append(node)
            right_node = node.right
            while(right_node != node):
                dl.cover_column(dl.columns[right_node.col])
                right_node = right_node.right
            if exact_cover_solution(solution, dl):
                return True
            solution.pop()
            left_node = node.left
            while(left_node != node):
                dl.uncover_column(dl.columns[left_node.col])
                left_node = left_node.left
            node = node.down
        dl.uncover_column(chosen_col)
   
    return False

def solve_sudoku(board):

    binary_matrix = create_binary_matrix(board) # function call to create binary matrix from the given sudoku grid
    dl = dancing_links(binary_matrix) # function call to convert binary matrix into dancing link structure 
    solution = []
    if not exact_cover_solution(solution, dl): # function call to solve the exact cover problem 
        print("No solution found")
        return
    N = len(board)
    solved_board = np.zeros((N, N), dtype=int)

    for node in solution:
        # converts the solution of dancing links into the sudoku solution 
        row, col = node.row // (N * N), (node.row % (N * N)) // N
        num = node.row % N + 1
        solved_board[row][col] = num

    return solved_board.tolist()

def sudoku_main_dlx(matrix):
    board = matrix.tolist()
    solved_board = solve_sudoku(board)
    
    # prints the solution 
    for row in solved_board:
        print(row)

