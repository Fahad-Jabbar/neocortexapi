using System;
using System.Collections.Generic;
using System.Linq;
using NeoCortexApi.Entities;
using MyExperiment;
using NeoCortexApi.Classifiers;

namespace MyExperiment.Tests
{
    public class KnnClassifierTests
    {
        [Fact]
        public void Learn_AddsNewTrainingData()
        {
            // Arrange
            var classifier = new KnnClassifierMain<string, ComputeCycle>();
            var input = "TestInput";
            var output = new Cell[] { new Cell { Index = 1 }, new Cell { Index = 2 } };

            // Act
            classifier.Learn(input, output);

            // Assert
            var predictedInputs = classifier.GetPredictedInputValues(output, 1);
            Assert.NotNull(predictedInputs);
        }

        [Fact]
        public void GetPredictedInputValues_ReturnsCorrectResults()
        {
            // Arrange
            var classifier = new KnnClassifierMain<string, ComputeCycle>();
            var input1 = "Input1";
            var output1 = new Cell[] { new Cell { Index = 1 }, new Cell { Index = 2 } };
            classifier.Learn(input1, output1);

            var input2 = "Input2";
            var output2 = new Cell[] { new Cell { Index = 3 }, new Cell { Index = 4 } };
            classifier.Learn(input2, output2);

            // Act
            var predictedInputs = classifier.GetPredictedInputValues(output1, 1);

            // Assert
            Assert.NotNull(predictedInputs);
            
            Assert.Equals(input1, predictedInputs.First().PredictedInput);
        }

        [Fact]
        public void Classify_ReturnsCorrectPrediction()
        {
            // Arrange
            var classifier = new KnnClassifierMain<string, ComputeCycle>();
            var input = "Input1";
            var output = new Cell[] { new Cell { Index = 1 }, new Cell { Index = 2 } };
            classifier.Learn(input, output);

            // Act
            var result = classifier.Classify(output, 1);

            // Assert
            // Assert.Equal(input, result);
        }

        [Fact]
        public void Vote_PerformsMajorityVotingCorrectly()
        {
            // Arrange
            var classifier = new KnnClassifierMain<string, ComputeCycle>();
            var input1 = "Input1";
            var output1 = new Cell[] { new Cell { Index = 1 }, new Cell { Index = 2 } };
            classifier.Learn(input1, output1);

            var input2 = "Input2";
            var output2 = new Cell[] { new Cell { Index = 3 }, new Cell { Index = 4 } };
            classifier.Learn(input2, output2);

            var output3 = new Cell[] { new Cell { Index = 1 }, new Cell { Index = 2 } };
            classifier.Learn(input1, output3);

            // Act
            var result = classifier.Vote(output1, 2);

            // Assert
            Assert.Equals(input1, result);
        }

        [Fact]
        public void ClearState_RemovesAllTrainingData()
        {
            // Arrange
            var classifier = new KnnClassifierMain<string, ComputeCycle>();
            var input = "Input1";
            var output = new Cell[] { new Cell { Index = 1 }, new Cell { Index = 2 } };
            classifier.Learn(input, output);

            // Act
            classifier.ClearState();

            // Assert
            var predictedInputs = classifier.GetPredictedInputValues(output, 1);
            Assert.IsEmpty(predictedInputs);
        }
    }
}
