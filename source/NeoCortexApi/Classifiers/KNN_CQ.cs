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
        private List<double[]> trainingData; // Stores the training data points.
        private List<int> labels; // Stores the labels corresponding to the training data points.
        private int k; // The number of nearest neighbors to consider for classification.

        // Static instance of Random class
        private static Random random = new Random();

        /// <summary>
        /// Initializes a new instance of the KNN_CQ class with the specified value of k.
        /// </summary>
        /// <param name="k">The number of nearest neighbors to consider for classification.</param>
        public KNN_CQ(int k)
        {
            this.k = k;
            trainingData = new List<double[]>();
            labels = new List<int>();
        }

        /// <summary>
        /// Generates a random Sparse Distributed Representation (SDR) with the specified dimensions and sparsity.
        /// </summary>
        /// <param name="dimensions">The number of dimensions of the SDR.</param>
        /// <param name="sparsity">The sparsity level of the SDR (proportion of active bits).</param>
        /// <returns>A randomly generated SDR.</returns>
        public static double[] GenerateRandomSdr(int dimensions, double sparsity)
        {
            var sdr = new double[dimensions];
            var activeColumns = (int)(sparsity * dimensions);

            for (int i = 0; i < activeColumns; i++)
            {
                sdr[random.Next(dimensions)] = 1.0;
            }

            return sdr;
        }

        /// <summary>
        /// Trains the KNN classifier with the provided training data and labels.
        /// </summary>
        /// <param name="data">The training data points.</param>
        /// <param name="targetLabels">The labels corresponding to the training data points.</param>
        public void Train(List<double[]> data, List<int> targetLabels)
        {
            if (data.Count != targetLabels.Count)
                throw new ArgumentException("Number of data points must be equal to the number of labels.");

            trainingData = data;
            labels = targetLabels;
        }

        /// <summary>
        /// Calculates the Euclidean distance between two data points.
        /// </summary>
        private double CalculateDistance(double[] point1, double[] point2)
        {
            if (point1.Length != point2.Length)
                throw new ArgumentException("Data points must have the same number of dimensions.");

            return Math.Sqrt(point1.Zip(point2, (x, y) => Math.Pow(x - y, 2)).Sum());
        }

        /// <summary>
        /// Predicts the label for a single data point using the trained KNN classifier.
        /// </summary>
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

        /// <summary>
        /// Predicts the label for a newly generated SDR and classifies it.
        /// </summary>
        public int ClassifyRandomSdr(int dimensions, double sparsity)
        {
            var dataPointToClassify = GenerateRandomSdr(dimensions, sparsity);
            return PredictSingleDataPoint(dataPointToClassify);
        }

        /// <summary>
        /// Sets the value of k, the number of nearest neighbors to consider for classification.
        /// </summary>
        /// <param name="k">The number of nearest neighbors to consider for classification.</param>
        public void SetK(int k)
        {
            if (k <= 0)
                throw new ArgumentException("k must be a positive integer.");

            this.k = k;
        }
    }
}
