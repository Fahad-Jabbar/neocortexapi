using System;
using System.Collections.Generic;
using System.Linq;

namespace KNN
{
    /// <summary>
    /// Represents a k-NN (k-Nearest Neighbor) classification program.
    /// </summary>
    public class KNNProgram
    {
        /// <summary>
        /// Main method to start the k-NN classification process.
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        public static void Main(string[] args)
        {
            Console.WriteLine("Starting the k-NN classification");

            var sequences = GetData();
            double[] testItem = new double[] { 1.0, 2.0, 3.0, 12.0 }; // Test item
            int k = 6; // Number of nearest neighbors to consider
            int classes = 2; // Number of classes

            Console.WriteLine("\nTraining Sequences:");
            foreach (var sequence in sequences)
            {
                Console.WriteLine($"{sequence.Key}: {string.Join(", ", sequence.Value)}");
            }

            Console.WriteLine($"\nNearest (k={k}) to test item: {string.Join(", ", testItem)}");
            string predictedSequence = Analyze(testItem, sequences, k, classes); // Perform k-NN classification
            Console.WriteLine($"\nPredicted Sequence: {predictedSequence}: {string.Join(", ", sequences[predictedSequence])}");

            Console.ReadLine();
        }



        /// <summary>
        /// Analyzes the test item to predict its sequence/class using k-NN classification.
        /// </summary>
        /// <param name="item">The item to be classified.</param>
        /// <param name="sequences">The training dataset.</param>
        /// <param name="k">The number of nearest neighbors to consider.</param>
        /// <param name="c">The number of classes.</param>
        /// <returns>The predicted sequence/class.</returns>

        public static string Analyze(double[] item, Dictionary<string, List<double>> sequences, int k, int c)
        {
            var distances = new Dictionary<string, double>(); // Dictionary to store distances
            foreach (var sequence in sequences)
            {
                double distance = EuclideanDistance(item, sequence.Value.ToArray());
                distances.Add(sequence.Key, distance);
            }

            var nearestSequences = distances.OrderBy(x => x.Value).Take(k); // Get the k nearest sequences

            Console.WriteLine("\nDistance to Nearest Sequences:");
            foreach (var seq in nearestSequences)
            {
                Console.WriteLine($"{seq.Key}: {seq.Value:F4}");
            }

            var votes = new Dictionary<string, double>(); // Dictionary to store votes for each sequence
            foreach (var seq in nearestSequences)
            {
                string sequenceName = seq.Key;
                votes[sequenceName] = 1.0 / seq.Value; // Weighted voting
            }

            Console.WriteLine("\nVoting Results:");
            foreach (var vote in votes)
            {
                Console.WriteLine($"{vote.Key}: {vote.Value:F4}");
            }

            string predictedSequence = votes.OrderByDescending(x => x.Value).First().Key; // Sequence with maximum votes
            CalculateMetrics(sequences, predictedSequence); // Calculate evaluation metrics

            return predictedSequence; // Return the predicted sequence
        }


        /// <summary>
        /// Calculates the Euclidean distance between two points.
        /// </summary>
        /// <param name="item">First point.</param>
        /// <param name="dataPoint">Second point.</param>
        /// <returns>Euclidean distance.</returns>

        static double EuclideanDistance(double[] item, double[] dataPoint)
        {
            double sum = 0.0;
            for (int i = 0; i < item.Length; ++i)
            {
                double diff = item[i] - dataPoint[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }


        /// <summary>
        /// Retrieves the training dataset.
        /// </summary>
        /// <returns>The training dataset.</returns>

        public static Dictionary<string, List<double>> GetData()
        {
            var sequences = new Dictionary<string, List<double>>();

            sequences.Add("S1", new List<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }));
            sequences.Add("S2", new List<double>(new double[] { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 }));

            return sequences;
        }


        /// <summary>
        /// Calculates evaluation metrics (accuracy, precision, recall, F1 score).
        /// </summary>
        /// <param name="sequences">The training dataset.</param>
        /// <param name="predictedSequence">The predicted sequence.</param>

        static void CalculateMetrics(Dictionary<string, List<double>> sequences, string predictedSequence)
        {
            int truePositives = 0, falsePositives = 0, falseNegatives = 0;
            foreach (var sequence in sequences)
            {
                string actualSequence = sequence.Key;
                if (actualSequence == predictedSequence)
                {
                    truePositives++; // Increment true positives
                }
                else
                {
                    falseNegatives++; // Increment false negatives
                    if (predictedSequence == actualSequence)
                        falsePositives++; // Increment false positives
                }
            }

            double accuracy = (double)truePositives / sequences[predictedSequence].Count; // Calculate accuracy
            double precision = (truePositives + falsePositives > 0) ? (double)truePositives / (truePositives + falsePositives) : 0; // Calculate precision
            double recall = (truePositives + falseNegatives > 0) ? (double)truePositives / (truePositives + falseNegatives) : 0; // Calculate recall
            double f1Score = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0; // Calculate F1 score

            Console.WriteLine($"\nAccuracy: {accuracy:F4}");
            Console.WriteLine($"Precision: {precision:F4}");
            Console.WriteLine($"Recall: {recall:F4}");
            Console.WriteLine($"F1 Score: {f1Score:F4}");
        }

    }
}
