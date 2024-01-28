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

        public KNNClassifier(int k)
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
    }
}

