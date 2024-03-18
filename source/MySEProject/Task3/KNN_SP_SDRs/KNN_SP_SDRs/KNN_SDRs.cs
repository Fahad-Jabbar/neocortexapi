using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApiSample
{
    public class KNNClassifier
    {
        private List<DataPoint> trainingData;

        public KNNClassifier(List<DataPoint> trainingData)
        {
            this.trainingData = trainingData;
        }

        public string Predict(DataPoint testDataPoint, int k)
        {
            List<DistanceLabelPair> distances = new List<DistanceLabelPair>();

            // Calculate distance from testDataPoint to each point in training data
            foreach (var dataPoint in trainingData)
            {
                double distance = CalculateDistance(testDataPoint, dataPoint);
                distances.Add(new DistanceLabelPair(distance, dataPoint.Label));
            }

            // Sort distances in ascending order
            distances.Sort((x, y) => x.Distance.CompareTo(y.Distance));

            // Take the k nearest neighbors
            var nearestNeighbors = distances.Take(k);

            // Count the occurrences of each label among the nearest neighbors
            Dictionary<string, int> labelCounts = new Dictionary<string, int>();
            foreach (var neighbor in nearestNeighbors)
            {
                if (labelCounts.ContainsKey(neighbor.Label))
                    labelCounts[neighbor.Label]++;
                else
                    labelCounts[neighbor.Label] = 1;
            }

            // Find the label with the highest count
            string predictedLabel = labelCounts.OrderByDescending(x => x.Value).First().Key;

            return predictedLabel;
        }

        private double CalculateDistance(DataPoint point1, DataPoint point2)
        {
            double sumOfSquares = 0;

            for (int i = 0; i < point1.Features.Length; i++)
            {
                sumOfSquares += Math.Pow(point1.Features[i] - point2.Features[i], 2);
            }

            return Math.Sqrt(sumOfSquares);
        }
    }

    public class DataPoint
    {
        public double[] Features { get; set; }
        public string Label { get; set; }
    }

    public class DistanceLabelPair
    {
        public double Distance { get; set; }
        public string Label { get; set; }

        public DistanceLabelPair(double distance, string label)
        {
            Distance = distance;
            Label = label;
        }
    }
}
