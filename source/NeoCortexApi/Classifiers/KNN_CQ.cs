using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi;
using NeoCortexApi.SpatialPooler;

/* This KNN Classifier is a prototype to test it on some randomly generated SDRs. Later, we are going to implement this using Neocortex API. 
 Once the model is predicting the desired outcome then we will feed the multisequence data to classify on new sequences.*/

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

        /// Generates a random Sparse Distribution Representation (SDR) data with the specified dimensions and sparsity.
        public static double[] GenerateRandomSdr(int dimensions, double sparsity)
        {
            var sdr = new double[dimensions];
            var activeColumns = (int)(sparsity * dimensions);

            for (int i = 0; i < activeColumns; i++)
            {
                sdr[Random.Shared.Next(dimensions)] = 1.0;
            }

            return sdr;
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

        /// Predicts the label for a single data point using the trained KNN classifier.
        public int PredictSingleDataPoint(double[] dataPointToClassify)
        {
            if (trainingData.Count == 0)
                throw new InvalidOperationException("The classifier has not been trained yet.");

            var distancesAndLabels = trainingData.Select((dataPoint, i) => new { Distance = CalculateDistance(dataPointToClassify, dataPoint), Label = labels[i] })
                                                  .OrderBy(x => x.Distance)
                                                  .Take(k)
                                                  .ToList();

            return distancesAndLabels.GroupBy(x => x.Label)
                                     .OrderByDescending(x => x.Count())
                                     .First()
                                     .Key;
        }

        /// Predicts the label for a newly generated SDR and classifies it.
        public int ClassifyRandomSdr(int dimensions, double sparsity)
        {
            var dataPointToClassify = GenerateRandomSdr(dimensions, sparsity);
            return PredictSingleDataPoint(dataPointToClassify);
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
