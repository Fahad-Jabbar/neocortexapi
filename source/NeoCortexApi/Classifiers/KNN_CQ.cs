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

    }
}

