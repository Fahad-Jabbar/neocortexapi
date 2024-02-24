using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApi.Classifiers
{ 

  /// Represents a k-Nearest Neighbors (KNN) classifier.

    public class KNN_CQ
    {
        private List<double[]> trainingData;
        private List<int> labels;
        private int k;

        /// Initializes a new instance of the KNN_CQ class with the specified value of k.
        public KNN_CQ(int k)
        {
            this.k = k;
            trainingData = new List<double[]>();
            labels = new List<int>();
        }
        
        /// Trains the KNN classifier with the provided training data and labels.
        public void Train(List<double[]> data, List<int> targetLabels)
        {
            if (data.Count != targetLabels.Count)
                throw new ArgumentException("Number of data points must be equal to the number of labels.");

            trainingData = data;
            labels = targetLabels;
        }

        /// Calculates the Euclidean distance between two data points.
        private double CalculateDistance(double[] point1, double[] point2)
        {
            if (point1.Length != point2.Length)
                throw new ArgumentException("Data points must have the same number of dimensions.");

            return Math.Sqrt(point1.Zip(point2, (x, y) => Math.Pow(x - y, 2)).Sum());
        }

        /// Predicts the labels for a list of new data points using the trained KNN classifier.
        public List<int> Predict(List<double[]> newData)
        {
            if (trainingData.Count == 0)
                throw new InvalidOperationException("The classifier has not been trained yet.");

            return newData.Select(PredictSingleInstance).ToList();
        }

        /// Predicts the label for a single new data point using the trained KNN classifier.
        private int PredictSingleInstance(double[] newDataPoint)
        {
            var distancesAndLabels = trainingData.Select((dataPoint, i) => new { Distance = CalculateDistance(newDataPoint, dataPoint), Label = labels[i] })
                                                 .OrderBy(x => x.Distance)
                                                 .Take(k)
                                                 .ToList();

            return distancesAndLabels.GroupBy(x => x.Label)
                                     .OrderByDescending(x => x.Count())
                                     .First()
                                     .Key;
        }

        /// Sets the value of k, the number of nearest neighbors to consider for classification.
        public void SetK(int k)
        {
            if (k <= 0)
                throw new ArgumentException("k must be a positive integer.");

            this.k = k;
        }
    }
}