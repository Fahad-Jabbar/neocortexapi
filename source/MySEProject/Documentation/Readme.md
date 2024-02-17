Project Title :- Implementation of KNN Classifier 
# KNN Classifier
-KNN (The k-nearest neighbors algorithm) is a popular machine learning algorithm used for classification and regression tasks.
-It's a type of instance-based learning where the algorithm makes predictions based on the majority class of the k-nearest neighbors of a given data point.
- A K-Nearest Neighbors (KNN) classifier is a type of instance-based learning algorithm used for classification tasks. 
- It classifies a data point based on the majority class of its k-nearest neighbors in the feature space. 
- The distance metric (usually Euclidean distance) is used to measure the similarity between data points.

# Class KNN_CQ.cs
Added a new class named KNN_CQ.cs for KNN Classifier modification, the class has different methods and a brief overview of the methods is given below:
- KNN_CQ(int k):
Constructor method that sets the KNN classifier's initial value to k, the number of closest neighbors to take into account when making predictions.
- Train(List<double[]> data, List<int> targetLabels):
A method for training the KNN classifier using the labels (targetLabels) and training data (data) that are supplied. 
- CalculateDistance(double[] point1, double[] point2):
Determines the Euclidean distance (between points 1 and 2) between two data points. The square root of the sum of squared differences is used to compute the distance after ensuring that both data points have the same number of dimensions.
- Predict(List<double[]> newData):
Uses the KNN algorithm to predict the labels for a list of fresh data points (newData). It computes the distances to every training data point, iteratively goes over each new data point, and uses majority vote to decide the expected label.
- PredictSingleInstance(double[] newDataPoint):
Uses the KNN algorithm to predict the label for a single new data point (newDataPoint). In order to establish the predicted label, it computes the distances between each fresh data point and each training data point, arranges the distances in ascending order, chooses the k-nearest neighbors, and uses majority vote.
- MajorityVoting(List<int> labels):
Uses a list of neighbor labels and majority voting to decide the expected label. Every label is counted, and the label with the greatest count is chosen as the projected label.