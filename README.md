# Sudoku-solver with image processing and multiple solving techniques---Backtracking-CPA-Dancing-Link

## Introduction
This repository presents a comprehensive solution for resolving Sudoku puzzles using advanced solving methods and image processing. A Sudoku picture is processed by the system, which then extracts and identifies the digits. One of four implemented algorithms—simple backtracking, CPA (Constraint Propagation Algorithm), DLX (Dancing Links), and an optimized DLX — can be used to solve the problem.

## Features
* Image Processing: Extracts individual cells from the given sudoku image.
  
* Digit Recognition: Uses a CNN model to recognize digits from processed cell images.
  
* Solving techniques: Includes simple backtracking, CPA, DLX or optimized DLX algorithm.
  
* Efficient Execution: Optimized DLX algorithm avoids the creation of binary matrix for faster operation.

## Image Processing
The image_processing.py script handles the preprocessing of the input Sudoku image. The steps involved are:

1. Reading the Image: The image is read using OpenCV.

2. Grayscale Conversion: The image is converted to grayscale to simplify further processing.
   
3. Thresholding: Adaptive thresholding is applied to binarize the image, enhancing the visibility of the grid lines.
   
4. Finding Contours: The contours in the image are detected to identify the largest box, which is assumed to be the outer boundary of the Sudoku grid.
   
5. Perspective Transformation: A perspective transform is applied to obtain a top-down view of the Sudoku grid.
    
6. Cell Extraction: The transformed grid is divided into 81 smaller images, each representing a single cell of the Sudoku puzzle.

## CNN Model for Digit Recognition
A Convolutional Neural Network (CNN) is used by the get_sudoku_matrix.py script to identify the digits included in the extracted cell images. The CNN model is designed to categorize numbers ranging from 1 to 9. Important facets of the model comprise of:

1. Architecture: The CNN consists of multiple convolutional layers, followed by max-pooling layers, and fully connected layers. This architecture is chosen for its effectiveness in image classification tasks. The architecture also implements the dropout technique to prevent overfitting.

2. Training Data: The model is trained on Char74K images dataset for 1-9 digits, further augmented to improve robustness. Some fonts that were far different from those that will actually appear in practical life were treated as outliers and omitted from the training.

3. Preprocessing: Each cell image is preprocessed to centralize the digit and normalize the pixel values before feeding it into the CNN for classification.

4. Digit Recognition: The trained CNN model predicts the digit for each cell image, with the output being a 9x9 matrix representing the Sudoku puzzle.

## Solving Algorithms
### Simple Backtracking
A straightforward depth-first search algorithm that tries all possibilities recursively. Suitable for smaller puzzles but can be considerably slow for complex ones.

### Constraint Propagation Algorithm (CPA)
Reduces the search spaces by using an arc-consistent(AC-3) algorithm and propagating constraints through forward checking inference, eliminating impossible values early in the solving process. Some heuristics like minimum remaining values with degree heuristic and least constraining value heuristic were used to determine which cell needs to be filled first with what value. 

### Dancing Link Algorithm (DLX)
An efficient algorithm based on Donald Knuth's Dancing Links technique, which is highly effective for exact cover problems like Sudoku. The sudoku grid was converted to a binary matrix representing cell constraint, row constraint, column constraint and box constraint. The binary matrix was then converted to a dancing link structure for solving. 

### Optimized Dancing Links (DLX)
Enhances the standard DLX by directly creating Dancing Links without an intermediate binary matrix, significantly improving execution speed.

## File Description
* sudoku_solver_main.py: Main script integrating all steps from image processing to solving the Sudoku puzzle.

* image_processing.py: Processes the input Sudoku image and splits it into 81 images, one for each cell.

* get_sudoku_matrix.py: Centralizes digits of each cell image and recognizes them using a pre-trained CNN model.

* sudoku_solver_backtracking.py: Implements the simple backtracking algorithm for solving Sudoku.

* sudoku_solver_cpa.py: Implements the Constraint Propagation Algorithm for solving Sudoku.

* sudoku_solver_dlx.py: Implements the DLX (Dancing Links) algorithm for solving Sudoku.

* sudoku_solver_dlx_optimized.py: Implements the optimized DLX algorithm without binary matrix creation.



