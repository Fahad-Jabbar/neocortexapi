using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeoCortexApi.Entities;
using System.Collections.Generic;
using NeoCortexApi.Classifiers;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Interface representing a classifier.
    /// </summary>
    public interface IClassifier
    {
        /// <summary>
        /// Calculates distances from a test data point to all points in the training data.
        /// </summary>
        /// <param name="testDataPoint">Data point for which distances need to be calculated.</param>
        /// <returns>List of distance-label pairs.</returns>
        List<DistanceLabelPair> CalculateDistances(DataPoint testDataPoint);

        /// <summary>
        /// Predicts the class label for a given test data point using the classifier algorithm.
        /// </summary>
        /// <param name="testDataPoint">Data point for which the class label needs to be predicted.</param>
        /// <param name="k">Number of nearest neighbors to consider.</param>
        /// <returns>Predicted class label for the test data point.</returns>
        string Predict(DataPoint testDataPoint, int k);
    }
}
