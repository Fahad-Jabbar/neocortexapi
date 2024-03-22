Project Title :- Implementation of KNN Classifier 
# KNN Classifier
- A K-Nearest Neighbors (KNN) classifier is a type of instance-based learning algorithm used for classification tasks. 
-It's a type of instance-based learning where the algorithm makes predictions based on the majority class of the k-nearest neighbors of a given data point.
- It classifies a data point based on the majority class of its k-nearest neighbors in the feature space. 
- The distance metric (Euclidean distance) is used to measure the similarity between data points.

The project consists of the following main parts:
- Implementing the native KNN classifier
- Integration of KNN with Neocortex API
- Using Spatial Pooler generated SDRs for KNN classifier.


##Implementation of Native KNN Classifier
The Native KNN classifier takes multiple data streams which belongs to three distinct classes based on two feature sets. Then an unknown data point (item)is fed to the classifier and Analyze(New data point, data stream, k value, No. of Classes) method is employed to predict the class to which the new data point belongs.

_**Sample Data**_

```
public static double[][] GetData()
 { 
   //Summary//
   //The Three Classes are 0, 1, 2//

     double[][] data = new double[30][];

     data[0] = new double[] { 0, 0.32, 0.43, 0 };
     data[1] = new double[] { 1, 0.26, 0.54, 0 };
     data[2] = new double[] { 2, 0.27, 0.6, 0 };
     .
     .
     .
     data[27] = new double[] { 27, 0.66, 0.14, 2 };
     data[28] = new double[] { 28, 0.64, 0.24, 2 };
     data[29] = new double[] { 29, 0.71, 0.22, 2 };

```
```
unclassified data point = { 0.38, 0.42 };
```
_**Path to the Native KNN**_
Inside `MySEProject` folder, there is a folder named `Task1`. From there run the `Program.cs` file to run this project.
Classifier: [Native KNN](https://github.com/Fahad-Jabbar/neocortexapi/blob/KNN_Quest1/source/MySEProject/Task1/KNN%20Classifier/KNN%20Classifier/Program.cs)

_**Testing**_
Unit tests are added for the KNN classifier under KNNTesting.cs
Path to the Unit test:[KNNTesting.cs] (https://github.com/Fahad-Jabbar/neocortexapi/blob/KNN_Quest1/source/MySEProject/Task1/KNN%20Classifier/KNN_Test/KNNTesting.cs)

## Implementation of KNN with Neocortex API
KNN classifier is implemented with Neocortex API, it takes multiple sequence with their respective labels and train the model. Once the model is trained, unknown data sequence can be tested on the model for classification.

**For Example:**
The unique integer sequences with labels are fed into the model for training. The KNN implementation of the classifier can then predict the unknown sequence using the Analyze(double[] item, Dictionary<string, List<double>> sequences, int k, int c) method.

_**Sample Data**_

```

1st Sequence = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }
2nd Sequence = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 }
3rd Sequence = { 14.0, 17.0, 20.0, 23.0, 27.0, 30.0, 33.0 }

unclassified Sequence = { 1.0, 2.0, 3.0, 7.0, 8.0, 9.0 }

```

_**Path to the Project**_
Inside `sample` folder, there is a folder named ` NeoCortexApiSample`. From there run the `Program.cs` file to run this project.
Classifier: [NneighborsClassifier.cs](https://github.com/Fahad-Jabbar/neocortexapi/blob/KNN_Quest1/source/NeoCortexApi/Classifiers/NneighborsClassifier.cs)

_**Testing**_
Unit tests are implemented for the KNN classifier in the KNN_UTest folder.
Path to the Unit test:[UnitTest1.cs] (https://github.com/Fahad-Jabbar/neocortexapi/blob/KNN_Quest1/source/KNN_UTest/UnitTest1.cs)


**Using SP generated SDRs for KNN Classifier:**
The KNN classifier integrated with the NeoCortex API for processing spatial pooler generated SDRs (Sparse Distributed Representations)is initiated by loading the csv file which consists of SDR data. During processing, the data is split into training and testing sets. 80% of the data is used for training and 20% is used as testing dataset. Once the model is trained, it can be tested on the testing dataset to classify the data.

The performance of the KNN classifier can also be evaluated through two distinct methods: k-fold cross-validation or a single test. In k-fold cross-validation, the dataset is divided into k equal-sized folds, and the classifier is trained and tested k times, with each fold serving as the testing set once and the remaining data used for training. This process allows for a comprehensive assessment of the classifier's generalization ability by ensuring that each data point is used for both training and testing at different stages. On the other hand, the single test option allows users to assess the classifier's performance on a single split of the dataset, typically divided into training and testing subsets using a the 80% to 20% ratio. This approach provides a quick evaluation of the classifier's accuracy without the computational overhead of k-fold cross-validation, making it suitable for scenarios where efficiency is a priority.

_**Sample Data**_

```

SDRs based Training Data = { 637,	641,	659,	661,	677,	704,	711,	738, 753,	761,	767,	786,	806,	848,	863,	931,	939,	947,	968,	1010,	0
184,	188,	202,	214,	216,	224,	225,	231,	252,	253,	267,	303,	310, 348,	352,	1975,	2010,	2026,	2028,	2029,	1}

unclassified Sequence = { 901, 936, 946, 953, 957, 961, 973, 981, 991, 997, 1002, 1014, 1017, 1025, 1034, 1047, 1069, 1100, 1111, 1128 }

```
The output of this project gives the labeled classification of the testing data.

_**Path to the Project**_
Inside `MySEProject` folder, there is a folder named `Task3`. From there run the `Program.cs` file to run this project.
[program.cs](https://github.com/Fahad-Jabbar/neocortexapi/blob/KNN_Quest1/source/MySEProject/Task3/KNN_SP_SDRs/KNN_SP_SDRs/KNN_SDRs.cs)

_**Testing**_
Unit tests are implemented for the KNN classifier in the KNN_UnitTest folder.
Path to the Unit test:[UnitTest1.cs] (https://github.com/Fahad-Jabbar/neocortexapi/blob/KNN_Quest1/source/MySEProject/Task3/KNN_SP_SDRs/KNN_UnitTest/UnitTest1.cs)



