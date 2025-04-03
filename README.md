# HW1_2210357124_idilsevis
 Implementation of k-Nearest Neighbors (k-NN) from scratch in Python as part of ELE 489: Fundamentals of Machine Learning
# k-NN Classifier on Wine Dataset

This project is a coursework assignment for ELE 489: Fundamentals of Machine Learning. It involves implementing the k-Nearest Neighbors (k-NN) algorithm from scratch and evaluating it on the Wine dataset** from the UCI Machine Learning Repository.

# Description
We build a custom k-NN classifier without using libraries like `sklearn.neighbors`. Our classifier supports both Euclidean and Manhattandistance metrics and can be tested with different `k` values.

#Files
- `knn.py`: Custom implementation of the k-NN classifier.
- `analysis.ipynb`: Jupyter notebook containing code, plots, and evaluation results.
- `README.md`: This file.

#Dataset
- UCI Wine dataset: https://archive.ics.uci.edu/dataset/109/wine
- It includes 178 samples, each with 13 numerical features and one class label (1, 2, or 3).

#Evaluation
- Accuracy is evaluated using different `k` values (1, 3, 5, 7, 9).
- Two distance metrics are tested: **Euclidean** and **Manhattan**.
- Visualizations include:
  - Feature relationships using `seaborn.pairplot`
  - Accuracy vs. K plot
  - Confusion matrix and classification report

#How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/knn-wine-classifier.git
   cd knn-wine-classifier
   ```
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```

