# Comparative Analysis of Adam, RMSprop, and AdamW Optimizers on a Feedforward Neural Network using the KMNIST Dataset

## Project Overview

This project compares the performance of three optimization algorithms—Adam, RMSprop, and AdamW—on a feedforward fully connected neural network using the KMNIST dataset. The KMNIST dataset consists of handwritten Japanese characters, providing a more complex challenge compared to the traditional MNIST dataset.

## Motivation
Selecting the right optimization algorithm is crucial for training deep neural networks efficiently. Different optimizers have unique behaviors in terms of convergence speed, stability, and generalization. This project aims to compare their performance based on accuracy, loss, and training time.

### Dataset
- **KMNIST Dataset**: A dataset of handwritten Japanese characters.
  - **Training Set**: 60,000 images
  - **Test Set**: 10,000 images
  - **Image Size**: 28x28 pixels, grayscale

### Model Architecture
- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layers**: Two hidden layers with 128 and 64 neurons respectively
- **Output Layer**: 10 neurons (one for each class)
- **Activation Functions**: ReLU for hidden layers, SoftMax for the output layer
- **Loss Function**: Cross-Entropy Loss

### Methodology
1. **Hyperparameter Tuning**: Systematic search for the best hyperparameters for each optimizer.
2. **Cross-Validation**: 5-fold cross-validation to ensure robust evaluation.
3. **Training and Evaluation**:
   - Train the model using each optimizer.
   - Evaluate performance on training, validation, and test datasets.
   - Record metrics such as accuracy, loss, and training time.

### Results
- **Tabular and Graphical Representation**:
  - Tables showing accuracy, loss, and training time for each optimizer.
  - Graphs comparing the performance metrics across different optimizers.

### Interpretation and Discussion
- **Analysis**: Discuss the performance of each optimizer, highlighting strengths and weaknesses.
- **Conclusion**: Summarize findings and suggest the best optimizer for this specific task.

## Repository Structure
```
kmnist-optimizer-comparison/
│── data/                  # Dataset (to be downloaded or generated)
│── notebooks/             # Jupyter notebooks for experiments
│── src/                   # Source code for model training & evaluation
│── results/               # Output files (graphs, logs, etc.)
│── README.md              # Project documentation
│── requirements.txt       # Dependencies for easy setup
│── .gitignore             # Ignore unnecessary files
```
## Getting Started
### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/davidabad98/kmnist-optimizer-comparison.git
   cd kmnist-optimizer-comparison
   ```
2. Download and prepare the KMNIST dataset.
3. Run the training script to start experiments.

## Contributors
- **[[[David Abad]](https://github.com/davidabad98)**
- **[[Rizvan Nahif]](https://github.com/joyrizvan)**
- **[[Darshil Shah]](https://github.com/darshil0811)**
- **[[[Navpreet Kaur Dusanje]](https://github.com/Navpreet-Kaur-Dusanje)**

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
KMNIST dataset provided by the [CODH](https://codh.rois.ac.jp/kmnist/).

