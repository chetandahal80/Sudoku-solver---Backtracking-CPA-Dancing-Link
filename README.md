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

