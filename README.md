# Email-Spam-detector
As an assignment for EECE 490 intro to machine learning course
Binary Classification with Neural Networks
This repository contains a comprehensive project on binary classification using neural networks, demonstrating the impact of various hyperparameter tuning techniques on model performance.

Overview
Binary classification is a supervised machine learning task where the model predicts one of two possible classes for input data. This project explores:

Building a dense neural network for binary classification.
Optimizing the network by adjusting hyperparameters, such as dropout rates, learning rates, batch sizes, and the number of neurons in hidden layers.
Evaluating model performance through metrics such as accuracy, loss, and confusion matrices.
Key Features
Neural Network Implementation:
Sequential API for building dense neural networks.
ReLU activation for hidden layers and Sigmoid activation for binary outputs.
Hyperparameter Tuning:
Adjusted dropout rates to optimize regularization.
Experimented with different learning rates, batch sizes, and epochs.
Explored the impact of increasing neurons in hidden layers.
Evaluation:
Plotted training/validation accuracy and loss.
Generated a confusion matrix to assess classification performance.
Provided detailed metrics (precision, recall, F1-score) using classification reports.
Skills Demonstrated
Deep learning and neural network architecture design.
Hyperparameter optimization for improved model performance.
Data preprocessing and model evaluation techniques.
Visualization of training metrics and confusion matrices.
Results
Best Configuration:
Batch size: 64
Learning rate: 0.0001
Dropout rate: 0.3
Epochs: 10
Achieved test accuracy: 98.90%
Insights:
Increasing neurons and reducing dropout rates improved performance.
A smaller learning rate with more epochs yielded the highest accuracy.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/binary-classification-neural-networks.git
Install the required Python libraries:
bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:
bash
Copy code
jupyter notebook binary_classification.ipynb
Usage
Open the notebook to explore step-by-step implementations and tuning experiments.
Train the model using the dataset and observe the impact of hyperparameter changes.
Evaluate the model using provided test data.
Files
binary_classification.ipynb: Jupyter Notebook with implementation and results.
README.md: Project overview and instructions.
Future Enhancements
Experiment with additional architectures, such as convolutional neural networks.
Incorporate advanced optimization techniques, like learning rate schedules.
Extend the model to handle multi-class classification tasks.
License
This project is licensed under the MIT License. See LICENSE for details.
