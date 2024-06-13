import numpy as np
from get_sudoku_matrix import get_matrix

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
    backtrack(matrix)
    
    

                    



