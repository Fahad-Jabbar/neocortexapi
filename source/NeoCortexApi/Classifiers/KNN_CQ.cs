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
    }
}
