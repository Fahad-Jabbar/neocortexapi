using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Class representing a K-Nearest Neighbors (KNN) classifier.
    /// </summary>
    public class KNNClassifier
    {
        private List<DataPoint> trainingData;

        /// <summary>
        /// Constructor for the KNNClassifier class.
        /// </summary>
        /// <param name="trainingData">List of data points used for training the classifier.</param>
        public KNNClassifier(List<DataPoint> trainingData)
        {
            this.trainingData = trainingData;
        }

        public List<DistanceLabelPair> CalculateDistances(DataPoint testDataPoint)
        {
            List<DistanceLabelPair> distances = new List<DistanceLabelPair>();

            // Calculate distance from testDataPoint to each point in training data
            foreach (var dataPoint in trainingData)
            {
                double distance = InternalDistance(testDataPoint, dataPoint);
                distances.Add(new DistanceLabelPair(distance, dataPoint.Label));
            }

            return distances;
        }

        // <summary>
        /// Calculates the Euclidean distance between two data points.
        /// </summary>
        /// <param name="point1">First data point.</param>
        /// <param name="point2">Second data point.</param>
        /// <returns>Euclidean distance between the two data points.</returns>
        public double InternalDistance(DataPoint point1, DataPoint point2)
        {
            double sumOfSquares = 0;

            for (int i = 0; i < point1.Features.Length; i++)
            {
                sumOfSquares += Math.Pow(point1.Features[i] - point2.Features[i], 2);
            }

            return Math.Sqrt(sumOfSquares);
        }

        /// <summary>
        /// Predicts the class label for a given test data point using KNN algorithm.
        /// </summary>
        /// <param name="testDataPoint">Data point for which the class label needs to be predicted.</param>
        /// <param name="k">Number of nearest neighbors to consider.</param>
        /// <returns>Predicted class label for the test data point.</returns>
        public string Predict(DataPoint testDataPoint, int k)
        {
            Dictionary<string, int> labelCounts = new Dictionary<string, int>();

            // Maintain a dictionary to store distances and corresponding labels
            Dictionary<double, string> distanceLabelMap = new Dictionary<double, string>();

            // Calculate distance from testDataPoint to each point in training data
            foreach (var dataPoint in trainingData)
            {
                double distance = InternalDistance(testDataPoint, dataPoint);

                // Store distance and label in the map
                distanceLabelMap[distance] = dataPoint.Label;
            }

            // Sort distances in ascending order
            var sortedDistances = distanceLabelMap.Keys.OrderBy(x => x);

            // Take the k nearest neighbors
            int count = 0;
            foreach (var distance in sortedDistances)
            {
                if (count >= k)
                    break;

                string label = distanceLabelMap[distance];

                // Update label counts
                if (labelCounts.ContainsKey(label))
                    labelCounts[label]++;
                else
                    labelCounts[label] = 1;

                count++;
            }

            // Find the label with the highest count
            string predictedLabel = labelCounts.OrderByDescending(x => x.Value).First().Key;

            return predictedLabel;
        }

        /// <summary>
        /// Calculates distances from a test data point to all points in the training data.
        /// </summary>
        /// <param name="testDataPoint">Data point for which distances need to be calculated.</param>
        /// <returns>List of distance-label pairs.</returns>
      
    }

    /// <summary>
    /// Class representing a data point with features and a label.
    /// </summary>
    public class DataPoint
    {
        public double[] Features { get; set; }
        public string Label { get; set; }
    }

    /// <summary>
    /// Class representing a pair of distance and corresponding label.
    /// </summary>
    public class DistanceLabelPair
    {
        public double Distance { get; }
        public string Label { get; }

        public DistanceLabelPair(double distance, string label)
        {
            Distance = distance;
            Label = label;
        }
    }
}
