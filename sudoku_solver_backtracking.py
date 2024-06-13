import numpy as np
from get_sudoku_matrix import get_matrix
import time


def feasible(matrix, x,y,n):
    # checks if the value is feasible to assign to the current cell or not 
    for i in range(9):
        # checks row
        if (matrix[x][i] == n and i != y):
            return False
    
    for i in range(9):
        # checks column
        if (matrix[i][y] == n and i != x):
            return False
    
    x0 = x//3
    y0 = y//3
    # checks the 3x3 box
    for i in range(x0*3, x0*3+3):
        for j in range(y0*3, y0*3+3):
            if(matrix[i][j] == n) and (i,j) != (x,y):
                return False

    return True

def backtrack(matrix):
    # backtracking algorithm
    for row in range(9):
        for col in range(9):
            if matrix[row][col] == 0:
                for n in range(1, 10):
                    if feasible(matrix, row, col, n):
                        matrix[row][col] = n
                        backtrack(matrix)
                        matrix[row][col] = 0
                return
    print (matrix)

def sudoku_main_backtrack(matrix):

    '''
    matrix = np.array([
        [8, 5, 0, 0, 0, 2, 4, 0, 0],
        [7, 2, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 7, 0, 0, 2],
        [3, 0, 5, 0, 0, 0, 9, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 7, 0],
        [0, 1, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 6, 0, 4, 0]
    ])'''
    '''matrix = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])'''
    '''matrix = np.array([[0, 0, 0, 0, 0, 9, 0, 7, 0],
    [7, 0, 6, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 6, 2, 0, 0],
    [6, 0, 7, 0, 0, 0, 0, 0, 4],
    [5, 0, 0, 0, 2, 0, 0, 0, 3],
    [0, 4, 0, 3, 5, 0, 1, 0, 0],
    [0, 0, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 0, 0]])'''

    start_time = time.time()
    backtrack(matrix)
    end_time = time.time()

    #print(f"Backtracking solver took {end_time - start_time:.6f} seconds")
    

                    



