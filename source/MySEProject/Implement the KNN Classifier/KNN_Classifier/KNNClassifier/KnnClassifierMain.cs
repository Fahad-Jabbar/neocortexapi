using NeoCortexApi.Classifiers;
using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MyExperiment
{
    // Generic K-nearest neighbors (KNN) classifier class
    public class KnnClassifierMain<TInput, TOutput> : IClassifierKnn<string, ComputeCycle>
    {
        // Dictionary for storing training data
        private readonly Dictionary<string, List<Cell[]>> _trainingData;
        private readonly short _k;  // Parameter for KNN

        // Constructor to initialize the training data dictionary and k value
        public KnnClassifierMain(short k = 3)
        {
            _trainingData = new Dictionary<string, List<Cell[]>>();
            _k = k;
        }

        // Method to add new training data
        public void Learn(string input, Cell[] output)
        {
            if (!_trainingData.ContainsKey(input))
            {
                _trainingData[input] = new List<Cell[]>();
            }

            _trainingData[input].Add(output);
        }

        // Method to get predicted input values based on KNN algorithm with parameter k
        public List<ClassifierResult<string>> GetPredictedInputValues(Cell[] predictiveCells, short k)
        {
            var distances = _trainingData.ToDictionary(
                entry => entry.Key,
                entry => CalculateMinimumDistance(predictiveCells, entry.Value)
            );

            var topKResults = distances.OrderBy(x => x.Value).Take(k)
                .Select(kvp => new ClassifierResult<string>
                {
                    PredictedInput = kvp.Key,
                    Similarity = 1.0 - kvp.Value
                })
                .ToList();

            return topKResults;
        }

        // Method to classify based on KNN algorithm with parameter k
        public string? Classify(Cell[] predictiveCells, short k = 1)
        {
            return GetPredictedInputValues(predictiveCells, k).FirstOrDefault()?.PredictedInput;
        }

        // Method to perform majority voting among KNN predictions with parameter k
        public string MajorityVoting(Cell[] predictiveCells, short k = 1)
        {
            var predictedInputs = GetPredictedInputValues(predictiveCells, k);

            var votes = predictedInputs.GroupBy(result => result.PredictedInput)
                                       .OrderByDescending(group => group.Count())
                                       .FirstOrDefault()
                                       ?.Key;

            return votes;
        }

        // Calculate minimum distance between predictive cells and training samples
        private double CalculateMinimumDistance(Cell[] input1, List<Cell[]> input2)
        {
            return input2.Min(trainingSample => CalculateHammingDistance(input1, trainingSample));
        }

        // Calculate Hamming distance between two cell arrays
        private double CalculateHammingDistance(Cell[] array1, Cell[] array2)
        {
            int lengthDifference = Math.Abs(array1.Length - array2.Length);
            int commonLength = Math.Min(array1.Length, array2.Length);

            int distance = lengthDifference + Enumerable.Range(0, commonLength)
                                                        .Count(i => array1[i].Index != array2[i].Index);

            return distance;
        }

        // Method to clear the state of the classifier (clear training data)
        public void ClearState() => _trainingData.Clear();
    }
}
