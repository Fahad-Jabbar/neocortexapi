using NUnit.Framework;
using System;
using System.Collections.Generic;

namespace KNN_UTest
{
    /// <summary>
    /// Unit tests for the k-NN (k-Nearest Neighbor) classification program.
    /// </summary>
    public class KNNTests
    {
        /// <summary>
        /// Test case for k-NN classification for a Test Sequence
        /// </summary>
        [Test]
        public void TestKNNClassificationForS1()
        {
            // Arrange
            // Define the training dataset with sequences S1 and S2
            var sequences = new Dictionary<string, List<double>>
            {
                { "S1", new List<double> { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 } },
                { "S2", new List<double> { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 } }
            };
            // Define the test item for sequence S1
            double[] testItem = new double[] { 1.5, 3.0, 4.0, 6.0 };
            int k = 3; // Number of nearest neighbors to consider
            int classes = 2; // Number of classes

            // Act
            // Perform k-NN classification
            var predictedSequence = KNN.KNNProgram.Analyze(testItem, sequences, k, classes);

            // Assert
            // Verify that the predicted sequence matches the expected sequence S1
            Assert.AreEqual("S1", predictedSequence);
            Console.WriteLine($"Predicted Sequence: {predictedSequence}");
        }

        /// <summary>
        /// Test case for k-NN classification for a different Test Sequence
        /// </summary>
        [Test]
        public void TestKNNClassificationForS2()
        {
            // Arrange
            // Define the training dataset with sequences S1 and S2
            var sequences = new Dictionary<string, List<double>>
            {
                { "S1", new List<double> { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 } },
                { "S2", new List<double> { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 } }
            };
            // Define the test item for sequence S2
            double[] testItem = new double[] { 10.5, 11.0, 12.5, 14.0 };
            int k = 3; // Number of nearest neighbors to consider
            int classes = 2; // Number of classes

            // Act
            // Perform k-NN classification
            var predictedSequence = KNN.KNNProgram.Analyze(testItem, sequences, k, classes);

            // Assert
            // Verify that the predicted sequence matches the expected sequence S2
            Assert.AreEqual("S2", predictedSequence);
            Console.WriteLine($"Predicted Sequence: {predictedSequence}");
        }
    }
}
