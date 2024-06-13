import os 
import time
from image_processing import obtain_sudoku_cells
from get_sudoku_matrix import get_matrix
from sudoku_solver_dlx_optimized import sudoku_main_dlx_optimized
from sudoku_solver_dlx import sudoku_main_dlx
from sudoku_solver_cpa import sudoku_main_cpa
from sudoku_solver_backtracking import sudoku_main_backtrack


start = time.time()
file_dir = os.path.dirname(os.path.abspath(__file__))
image_file = os.path.join(file_dir, "sud.png") # path for sudoku image
destination_file = os.path.join(file_dir, "sudoku_cells") # destination path to store each cells of sudoku after splitting
os.makedirs(destination_file, exist_ok=True) 

obtain_sudoku_cells(image_file, destination_file) # function call to extract each cell from the sudoku image
matrix = get_matrix(destination_file) # function call to convert sudoku image into matrix 
print("Sudoku matrix =")
print(matrix)

sudoku_main_dlx_optimized(matrix) # function call to solve the sudoku using optimized dancing link algorithm 
#sudoku_main_dlx(matrix) # function call to solve the sudoku using dancing link algorithm 
#sudoku_main_cpa(matrix) # function call to solve the sudoku using constraint propagation algorithm
#sudoku_main_backtrack(matrix) # function call to solve the sudoku using simple backtracking algorithm

end = time.time()

print(f"Took {(end - start):.6f} to solve the sudoku puzzle")
