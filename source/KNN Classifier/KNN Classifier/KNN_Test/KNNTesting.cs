using NUnit.Framework;
using KNN;

namespace KNN.nUnitTests
{
    [TestFixture]
    public class KNNTesting
    {
        [Test]
        public void Analyze_ValidInput_ReturnsCorrectPredictedClass1()
        {
            // Arrange
            double[][] data = KNNProgram.GetData();
            double[] item = new double[] { 0.38, 0.42 };
            int k = 6;
            int c = 3;

            // Act
            int predictedClass = KNNProgram.Analyze(item, data, k, c);

            // Assert
            Assert.AreEqual(1, predictedClass); // Assuming the expected predicted class is 1
        }

        [Test]
        public void Analyze_ValidInput_ReturnsCorrectPredictedClass2()
        {
            // Arrange
            double[][] data = KNNProgram.GetData();
            double[] item = new double[] { 0.63, 0.25 };
            int k = 6;
            int c = 3;

            // Act
            int predictedClass = KNNProgram.Analyze(item, data, k, c);

            // Assert
            Assert.AreEqual(2, predictedClass); // Assuming the default predicted class is 2
        }

        [Test]
        public void Analyze_ValidInput_ReturnsCorrectPredictedClass3()
        {
            // Arrange
            double[][] data = KNNProgram.GetData();
            double[] item = new double[] { 0.50, 0.30 };
            int k = 6;
            int c = 3;

            // Act
            int predictedClass = KNNProgram.Analyze(item, data, k, c);

            // Assert
            Assert.AreEqual(0, predictedClass); // Assuming the default predicted class is 0
        }

        [Test]
        public void Analyze_ValidInput_ReturnsCorrectPredictedClass4()
        {
            // Arrange
            double[][] data = KNNProgram.GetData();
            double[] item = new double[] { 0.40, 0.40 };
            int k = 6;
            int c = 3;

            // Act
            int predictedClass = KNNProgram.Analyze(item, data, k, c);

            // Assert
            Assert.AreEqual(1, predictedClass); // Assuming the default predicted class is 1
        }
    }
}
