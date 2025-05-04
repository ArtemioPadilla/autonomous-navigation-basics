# autonomous-navigation-basics
Basics of Autonomous Navigation

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── notebooks/       # Jupyter Notebooks for tutorials and experiments
│   │   └── 1-hough-transform.ipynb  # Notebook demonstrating Hough Transform
│   └── ...             # Other code files and scripts
├── docs/
│   ├── books/          # Books and chapters in PDF and DOCX formats
│   ├── essays/         # Essays in Markdown (English and Spanish)
│   ├── papers/         # Research papers in PDF format
│   └── readme.md       # Documentation for the docs folder
└── rsc/
    └── samples/        # Sample images and resources
        └── sudoku.png  # Example image used in the Hough Transform notebook
```

## Notebooks

### 1. Hough Transform

- **File**: `code/notebooks/1-hough-transform.ipynb`
- **Description**: This notebook demonstrates the use of the Hough Transform for line detection in images. It includes examples of both the Standard Hough Transform and the Probabilistic Hough Transform.
- **Sample Image**: The notebook uses a Sudoku puzzle image located at `rsc/samples/sudoku.png`.
- **Key Features**:
  - Explains the theory behind the Hough Transform.
  - Demonstrates OpenCV's implementation of the Standard and Probabilistic Hough Transforms.
  - Visualizes the detected lines on the sample image.

## Requirements

To run the notebooks and scripts in this repository, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/autonomous-navigation-basics.git
   cd autonomous-navigation-basics
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the `code/notebooks/` directory and open the desired notebook using Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook 1-hough-transform.ipynb
   ```

4. Follow the instructions in the notebook to run the examples and visualize the results.

## Resources

- **Sample Image**: The Sudoku puzzle image used in the Hough Transform notebook is located at `rsc/samples/sudoku.png`.
- **Documentation**: Additional documentation and resources can be found in the `docs/` folder.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.