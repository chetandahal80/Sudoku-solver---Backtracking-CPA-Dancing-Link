import numpy as np
from get_sudoku_matrix import get_matrix
import time
import cProfile
import pstats

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
    def __init__(self, N):
        self.header = columnHeader(-1)
        self.columns = []

        # creating circular doubly linked list between all column headers
        for col in range(4*N*N):
            new_column = columnHeader(col)
            self.columns.append(new_column)
            self.header.left.right = new_column
            new_column.left = self.header.left
            new_column.right = self.header
            self.header.left = new_column
    
    # creating circular doubly linked list between all the nodes in the dancing link
    def add_row(self, constraint, row):
        prev_node = None
        row_header = None
        for col_index in constraint:
            new_node = Node(row, col_index)
            column_header = self.columns[col_index]

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

    def print_structure(self):
        # This is a utility function to print the dancing links structure
        for col_header in self.columns:
            print(f"Column {col_header.col}, Size: {col_header.size}")
        
# returns the column number to place 1 in the respecting row of dancing link structure 
def cover_cell(N, n_row, n_col, num):
    row = [n_row*N + n_col, N*N + n_row*N + num - 1, 2*N*N + n_col*N + num - 1, 3*N*N + (n_row//3 * 3 + n_col//3)*N + num - 1]
    return row

def store_constraint(board):
    N = len(board)
    constraint = []
    dl = dancing_links(N) # creates dancing link object
    for i in range(N):
        for j in range(N):
            if (board[i][j] != 0):
                # adds row to dancing link structure directly based on the cell value of the sudoku grid
                constraint = cover_cell(N, i, j, board[i][j])
                row = (i*N+j)*N + board[i][j] - 1
                dl.add_row(constraint, row)
            else:
                for num in range(1,N+1):
                    # adds nine rows to the dancing link structure for empty sudoku cell 
                    constraint = cover_cell(N, i, j, num)
                    row = (i*N+j)*N + num - 1
                    dl.add_row(constraint, row)

    return dl

# finds the exact cover solution using dancing link algorithm (Algorithm X)
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
    # creates dancing link structure directly without creating binary matrix
    dl = store_constraint(board) # function call to create dancing link structure
    solution = []
    
    if not exact_cover_solution(solution, dl):
        print("No solution found")
        return
    
    N = len(board)
    solved_board = np.zeros((N, N), dtype=int)

    for node in solution:
        # converts solution of dancing link algorithm to sudoku solution 
        row, col = node.row // (N * N), (node.row % (N * N)) // N
        num = node.row % N + 1
        solved_board[row][col] = num

    return solved_board.tolist()


def sudoku_main_dlx_optimized(matrix):

    board = matrix.tolist()
    solved_board = solve_sudoku(board)
    
    # printing solution 
    for row in solved_board:
        print(row)

   

