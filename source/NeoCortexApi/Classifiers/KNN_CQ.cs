using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApi.Classifiers
{ 
   /// <summary>
  /// Represents a k-Nearest Neighbors (KNN) classifier.
  /// </summary>
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

        private double CalculateDistance(double[] point1, double[] point2)
        {
            if (point1.Length != point2.Length)
                throw new ArgumentException("Data points must have the same number of dimensions.");

            return Math.Sqrt(point1.Zip(point2, (x, y) => Math.Pow(x - y, 2)).Sum());
        }

        public List<int> Predict(List<double[]> newData)
        {
            if (trainingData.Count == 0)
                throw new InvalidOperationException("The classifier has not been trained yet.");

            return newData.Select(PredictSingleInstance).ToList();
        }

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

        public void SetK(int k)
        {
            if (k <= 0)
                throw new ArgumentException("k must be a positive integer.");

            this.k = k;
        }
    }
}