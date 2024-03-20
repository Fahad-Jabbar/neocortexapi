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
            Dictionary<string, int> labelCounts = new Dictionary<string, int>();

            // Maintain a dictionary to store distances and corresponding labels
            Dictionary<double, string> distanceLabelMap = new Dictionary<double, string>();

            // Calculate distance from testDataPoint to each point in training data
            foreach (var dataPoint in trainingData)
            {
                double distance = CalculateDistance(testDataPoint, dataPoint);

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

        public List<DistanceLabelPair> CalculateDistances(DataPoint testDataPoint)
        {
            List<DistanceLabelPair> distances = new List<DistanceLabelPair>();

            // Calculate distance from testDataPoint to each point in training data
            foreach (var dataPoint in trainingData)
            {
                double distance = CalculateDistance(testDataPoint, dataPoint);
                distances.Add(new DistanceLabelPair(distance, dataPoint.Label));
            }

            return distances;
        }

        public double CalculateDistance(DataPoint point1, DataPoint point2)
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
        public double Distance { get; }
        public string Label { get; }

        public DistanceLabelPair(double distance, string label)
        {
            Distance = distance;
            Label = label;
        }
    }
}
