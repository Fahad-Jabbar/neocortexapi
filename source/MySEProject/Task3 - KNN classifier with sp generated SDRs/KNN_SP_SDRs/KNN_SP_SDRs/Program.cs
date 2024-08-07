using NeoCortexApi;
using NeoCortexApiSample;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace MyExperiment
{
    internal class Program
    {
        /// <summary>
        /// Main entry point for the program.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            ExecuteSequenceLearningExperiment();
        }

        /// <summary>
        /// Executes the sequence learning experiment.
        /// </summary>
        private static void ExecuteSequenceLearningExperiment()
        {
            var sequences = new Dictionary<string, List<double>>
            {
                { "Sequence1", new List<double> { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 } },
                { "Sequence2", new List<double> { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 } }
            };

            var experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);

            EvaluatePredictions(predictor, sequences["Sequence1"].ToArray());
            EvaluatePredictions(predictor, sequences["Sequence2"].ToArray());
        }

        /// <summary>
        /// Evaluates the predictions for a given sequence.
        /// </summary>
        /// <param name="predictor">The predictor instance.</param>
        /// <param name="sequence">The sequence to predict.</param>
        private static void EvaluatePredictions(Predictor predictor, double[] sequence)
        {
            Debug.WriteLine(new string('-', 30));

            foreach (var value in sequence)
            {
                var predictions = predictor.Predict(value);

                if (predictions.Count > 0)
                {
                    foreach (var prediction in predictions)
                    {
                        Debug.WriteLine($"{prediction.PredictedInput} - {prediction.Similarity}");
                    }

                    var sequenceInfo = predictions.First().PredictedInput.Split(new[] { '_', '-' }, StringSplitOptions.RemoveEmptyEntries);
                    Debug.WriteLine($"Predicted Sequence: {sequenceInfo[0]}, Predicted Next Element: {sequenceInfo.Last()}");
                }
                else
                {
                    Debug.WriteLine("No predictions available :(");
                }
            }

            Debug.WriteLine(new string('-', 30));
        }
    }
}
