# BERT Text Classification for Clickbait Detection

This repository provides an overview of a project where I implemented a **BERT-based text classification model** to detect clickbait. The project involved fine-tuning a pre-trained BERT model and optimizing hyperparameters to maximize classification performance.

---

## Project Overview

The goal of this project was to build a **text classification model** using BERT to distinguish between clickbait and non-clickbait headlines. The model was trained and evaluated with a focus on optimizing performance metrics, including precision, recall, and accuracy. The project emphasized hyperparameter tuning and experimentation to achieve the best results.

---

## Key Features

1. **BERT Implementation**:
   - Used the **Transformers** library to fine-tune a pre-trained BERT model for the specific task of clickbait detection.

2. **Hyperparameter Tuning**:
   - Experimented with the following hyperparameters to optimize model performance:
     - **`num_epochs`**: Adjusted the number of training epochs to balance learning and overfitting.
     - **`batch_size`**: Tuned for efficient training and memory management.
     - **`optimizer_cls`**: Tested different optimizers (e.g., AdamW).
     - **`learning_rate`**: Carefully adjusted the learning rate for smooth and effective optimization.
     - **`weight_decay`**: Applied weight decay to regularize the model and prevent overfitting.

3. **Metrics for Evaluation**:
   - Calculated the following metrics to evaluate model performance:
     - **True Positives (TP)**: Correctly identified clickbait.
     - **True Negatives (TN)**: Correctly identified non-clickbait.
     - **False Positives (FP)**: Incorrectly labeled non-clickbait as clickbait.
     - **False Negatives (FN)**: Missed identifying clickbait headlines.
   - Used these metrics to derive precision, recall, and F1 scores.

4. **Data Visualization**:
   - Leveraged **Matplotlib** to visualize training and validation loss, accuracy, and other performance trends.

5. **PyTorch Integration**:
   - Utilized **PyTorch** for training, validation, and implementation of the BERT model, including custom data loaders and training loops.

---

## Implementation Details

1. **Model Training**:
   - Fine-tuned the pre-trained BERT model on a dataset of headlines labeled as clickbait or non-clickbait.
   - Implemented a training loop to optimize the model using backpropagation and gradient descent.

2. **Hyperparameter Experiments**:
   - Conducted experiments by varying `num_epochs`, `batch_size`, and `learning_rate` to identify the optimal configuration.
   - Tested different settings for weight decay and optimizer to improve generalization.

3. **Evaluation Pipeline**:
   - Split the dataset into training, validation, and test sets.
   - Measured performance on the validation set to tune hyperparameters and tested the final model on unseen data.

---

## Libraries and Tools

- **Transformers**: For accessing and fine-tuning the pre-trained BERT model.
- **PyTorch**: For model implementation, training, and optimization.
- **Matplotlib**: For visualizing training and evaluation metrics.

---

## Acknowledgments
A special thanks to the instructors of my NLP course, Professors David Mortensen and Eric Nyberg, for their invaluable guidance on this project.

---

For any questions, feel free to reach out to **anirudhm@andrew.cmu.edu**!
