import numpy as np
from get_sudoku_matrix import get_matrix

def solve_sudoku(matrix):
    variables, domains, constraints = convert_to_csp(matrix) # function call to convert given problem into CSP 
    if not ac3(domains, constraints): # making the problem arc consistent
        return None
    return backtrack({}, variables, domains, constraints) # function call to backtracking algorithm 

def backtrack(assignment, variables, domains, constraints):
    # backtracking algorithm 
    if set(assignment.keys()) == set(variables):
        return assignment
    var = select_unassigned_variable(variables, domains, assignment, constraints) # MRV with degree heuristic
    for value in order_domain_values(var, domains, constraints): # least constraining value heuristic
        if is_consistent(var, value, assignment, constraints):
            assignment[var] = value
            domains_copy = {k: v.copy() for k, v in domains.items()}
            if inference(domains, constraints, var, value): # inference for constraint propagation 
                result = backtrack(assignment, variables, domains, constraints)
                if result is not None:
                    return result
            assignment.pop(var)
            domains = domains_copy
    
    return None

def is_consistent(var, value, assignment, constraints):
    # checks if a current value assignemnt to a cell is consistent with other cells in the assignment dictionary or not
    for neighbour in neighbours(var, constraints):
        if neighbour in assignment:
            if value == assignment[neighbour]:
                return False
    return True

def select_unassigned_variable(variables, domains, assignment,constraints):
    # selects unassigned variable based on minimum remaining value (MRV) heuristic combined with degree heuristic
    unassigned_vars = [v for v in variables if v not in assignment]
    return min(unassigned_vars, key=lambda var: (len(domains[var]), -len(neighbours(var, constraints))))

def order_domain_values(var, domains, constraints):
    # arranges the domain of a variable based on number of conflicts with neighbour cells [Least Constraining Value heuristic]
    return sorted(domains[var], key=lambda val: num_conflicts(var, val, domains, constraints))

def num_conflicts(var, value, domains, constraints):
    # calculates the number of neighbour cells that are in conflict with a value assigned to a cell
    return sum(1 for neighbor in neighbours(var, constraints) if value in domains[neighbor])

def ac3(domains, constraints):
    # Arc consistency algorithm 
    queue = [(Xi, Xj) for (Xi, Xj) in constraints]
    while queue:
        Xi, Xj = queue.pop(0)
        if revise(Xi, Xj, domains):
            if not domains[Xi]:
                return False
            for Xk in neighbours(Xi, constraints):
                if Xk != Xj:
                    queue.append((Xk, Xi))
    return True

def revise(Xi, Xj, domains):
    # removes any domain value that makes a cell arc inconsistent
    revised = False
    for x in domains[Xi].copy():
        if not any(x != y for y in domains[Xj]):
            domains[Xi].remove(x)
            revised = True
    return revised

def inference(domains, constraints, var, value):
    # performs inferences (forward checking) after a variable is assigned a value 
    # removes a value from domain of neighbour cells if the constraint is violated 
    for neighbor in neighbours(var, constraints):
        if value in domains[neighbor]:
            domains[neighbor].remove(value)
            if not domains[neighbor]:
                return False
    return True

def neighbours(var, constraints):
    # returns the cells which share constraints with the cell indicated by "var"
    return {Xi if Xj == var else Xj for (Xi, Xj) in constraints if var in (Xi, Xj)}

def convert_to_csp(matrix):
    # converts the given sudoku matrix into constraint satisfying problem (CSP)
    # takes partially filled sudoku matrix as input and returns variables, domains and constraints separately

    variables = []
    domains = {}
    units = []
    constraints = []
    
    # filling variables list and domains dictionary
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                variables.append((i, j))
                domains[(i, j)] = set(range(1, 10))
            else:
                domains[(i, j)] = {matrix[i][j]}

    # filling constraints list
    for i in range(9):
        units.append([(i, j) for j in range(9)])
        units.append([(j, i) for j in range(9)])
    
    for i in range(3):
        for j in range(3):
            units.append([(i * 3 + x, j * 3 + y) for x in range(3) for y in range(3)])

    # creats list containt tuple of each cell and its each neigbour cells
    for unit in units:
        constraints += [(cell1, cell2) for cell1 in unit for cell2 in unit if cell1 != cell2]

    return variables, domains, constraints

def print_board(board):
    # prints solution in matrix form
    for row in board:
        print(row)


def sudoku_main_cpa(matrix):
    board = matrix.tolist()
    solution = solve_sudoku(board)
    
    # print solution 
    if solution:
        for (i, j), value in solution.items():
            board[i][j] = value
        print_board(board)
    else:
        print("No solution found")
