using System;
using System.Collections.Generic;
using System.Text;

namespace NeoCortexApi.Classifiers
{
    internal class KNN_CQ
    {
        private List<double[]> trainingData;
        private List<int> labels;
        private int k;

        public KNN_CQ(int k)
        {
            this.k = k;
            trainingData = new List<double[]>();
            labels = new List<int>();
        }
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

            double sumOfSquares = 0.0;

            for (int i = 0; i < point1.Length; i++)
            {
                double diff = point1[i] - point2[i];
                sumOfSquares += diff * diff;
            }

            return Math.Sqrt(sumOfSquares);
        }

        public List<int> Predict(List<double[]> newData)
        {
            if (trainingData.Count == 0)
                throw new InvalidOperationException("The classifier has not been trained yet.");

            List<int> predictedLabels = new List<int>();

            foreach (var dataPoint in newData)
            {
                int predictedLabel = PredictSingleInstance(dataPoint);
                predictedLabels.Add(predictedLabel);
            }

            return predictedLabels;
        }

        // Adding a method for distance sorting

        private int PredictSingleInstance(double[] newDataPoint)
        {
            // Calculate distances from newDataPoint to all training data points
            List<Tuple<double, int>> distancesAndLabels = new List<Tuple<double, int>>();

            for (int i = 0; i < trainingData.Count; i++)
            {
                double distance = CalculateDistance(newDataPoint, trainingData[i]);
                distancesAndLabels.Add(new Tuple<double, int>(distance, labels[i]));
            }

            // Sort the distancesAndLabels list based on distances in ascending order
            distancesAndLabels.Sort((x, y) => x.Item1.CompareTo(y.Item1));

            // Take the first k elements from the sorted list
            List<int> kNearestLabels = new List<int>();

            for (int i = 0; i < k; i++)
            {
                kNearestLabels.Add(distancesAndLabels[i].Item2);
            }

            // Perform a majority voting to determine the predicted label
            int predictedLabel = MajorityVoting(kNearestLabels);

            return predictedLabel;
        }

        private int MajorityVoting(List<int> labels)
        {
            Dictionary<int, int> labelCount = new Dictionary<int, int>();

            foreach (var label in labels)
            {
                if (labelCount.ContainsKey(label))
                    labelCount[label]++;
                else
                    labelCount[label] = 1;
            }

        }
}
    // Add a counting method
    private int MajorityVoting(List<int> labels)
    {
        Dictionary<int, int> labelCount = new Dictionary<int, int>();

        foreach (var label in labels)
        {
            if (labelCount.ContainsKey(label))
                labelCount[label]++;
            else
                labelCount[label] = 1;
        }
