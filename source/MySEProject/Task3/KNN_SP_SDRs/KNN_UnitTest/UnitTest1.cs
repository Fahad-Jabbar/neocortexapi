using NUnit.Framework;
using NeoCortexApiSample;
using System;
using System.Collections.Generic;

namespace KNN_UnitTest
{
    public class Tests
    {
        private List<DataPoint> trainingData;
        private KNNClassifier knn;

        /// <summary>
        /// Sets up the test environment by initializing training data and the KNN classifier.
        /// </summary>
        [SetUp]
        public void Setup()
        {
            // Initialize training data with sample data points
            trainingData = new List<DataPoint>
            {
                // Sample training data for label 'X'
                new DataPoint { Features = new double[] { 800, 850, 890, 920, 930, 940, 970, 980, 990, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1120 }, Label = "X" },
                // Sample training data for label 'Y'
                new DataPoint { Features = new double[] { 920, 950, 980, 1000, 1010, 1020, 1050, 1060, 1080, 1090, 1100, 1120, 1130, 1150, 1160, 1170, 1180, 1190, 1200, 1220 }, Label = "Y" },
                // Add more training data as needed
            };

            // Initialize KNN classifier with the training data
            knn = new KNNClassifier(trainingData);
        }

        /// <summary>
        /// Tests the classification process using a specific SDR input for label 'X'.
        /// This test verifies whether the KNN classifier correctly predicts the label 'X' based on the provided SDR input.
        /// </summary>
        [Test]
        public void TestClassificationWithSDR1()
        {
            // Define SDR input for classification
            double[] sdrInput = { 900, 940, 950, 960, 970, 980, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130 };

            // Predict the class label using the KNN classifier and assert the result
            string predictedLabel = knn.Predict(new DataPoint { Features = sdrInput }, 1);
            Console.WriteLine($"Test Case 1: Expected: X, Actual: {predictedLabel}");
            Assert.AreEqual("X", predictedLabel);
        }

        /// <summary>
        /// Tests the classification process using a specific SDR input for label 'Y'.
        /// This test verifies whether the KNN classifier correctly predicts the label 'Y' based on the provided SDR input.
        /// </summary>
        [Test]
        public void TestClassificationWithSDR2()
        {
            // Define another SDR input for classification
            double[] sdrInput = { 920, 960, 980, 990, 1000, 1020, 1060, 1070, 1080, 1090, 1100, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200 };

            // Predict the class label using the KNN classifier and assert the result
            string predictedLabel = knn.Predict(new DataPoint { Features = sdrInput }, 1);
            Console.WriteLine($"Test Case 2: Expected: Y, Actual: {predictedLabel}");
            Assert.AreEqual("Y", predictedLabel);
        }
    }
}
