using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeoCortexApiSample;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using System.Diagnostics;
using System.ComponentModel;
using System.Security.Cryptography.X509Certificates;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Data;

namespace NeoCortexApiSample
{
    class Program
    {
        /// <summary>
        /// Main entry point of the program.
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        static void Main(string[] args)
        {

            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            //SpatialPatternLearning experiment = new SpatialPatternLearning();
            //experiment.Run();

            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            // SequenceLearning experiment = new SequenceLearning();
            // experiment.Run();

            RunMultiSimpleSequenceLearningExperiment();
            RunMultiSequenceLearningExperiment();

            Console.WriteLine("Classification with KNN!");

            // Load data from CSV file
            List<DataPoint> data = LoadData("Dataset.csv");
            Console.WriteLine($"Dataset loaded with {data.Count} data points.");

            // Display sample data points
            DisplaySampleData(data);

            // Define value of k
            int k = GetKFromUser();

            // Ask the user whether to perform k-fold cross-validation or single test
            Console.WriteLine("Choose an option:");
            Console.WriteLine("1. Perform k-fold cross-validation");
            Console.WriteLine("2. Test the classifier once");
            Console.Write("Enter your choice (1 or 2): ");

            int choice;
            while (!int.TryParse(Console.ReadLine(), out choice) || (choice != 1 && choice != 2))
            {
                Console.WriteLine("Invalid input. Please enter 1 or 2.");
            }

            if (choice == 1)
            {
                // Perform k-fold cross-validation
                int numFolds = 5; // You can adjust the number of folds as needed
                Console.WriteLine($"Performing {numFolds}-fold cross-validation...");
                double[] accuracies = CrossValidate(data, k, numFolds);
                double averageAccuracy = accuracies.Sum() / numFolds;
                Console.WriteLine($"Average accuracy over {numFolds} folds: {averageAccuracy}%");
            }
            else
            {

                // Test the classifier once for accuracy 
                Console.WriteLine("Testing the classifier once...");
                List<DataPoint> trainingData, testingData;
                (trainingData, testingData) = SplitData(data, 0.8); // 80% training data, 20% testing data
                KNNClassifier knn = new KNNClassifier(trainingData);

                // Test the classifier on testing dataset
                double accuracy = TestClassifier(knn, testingData, k);
                Console.WriteLine($"Accuracy: {accuracy}%");

                // Add new data point for prediction
                double[] newData = { 901, 936, 946, 953, 957, 961, 973, 981, 991, 997, 1002, 1014, 1017, 1025, 1034, 1047, 1069, 1100, 1111, 1128 };
                DataPoint newDataPoint = new DataPoint { Features = newData };

                // Display distances from k nearest data points in nearest to farthest order
                Console.WriteLine("Distances from K Nearest Data Points:");
                List<DistanceLabelPair> distances = knn.CalculateDistances(newDataPoint);
                foreach (var pair in distances.Take(k))
                {
                    Console.WriteLine($"  Distance: {pair.Distance,-10:F2} Label: {pair.Label}");
                }

                // Calculate sums for both labels among k nearest neighbors
                int sumLabel0 = distances.Take(k).Count(d => d.Label == "0");
                int sumLabel1 = distances.Take(k).Count(d => d.Label == "1");
                Console.WriteLine($"Sum of Numbers for Label 0: {sumLabel0}");
                Console.WriteLine($"Sum of Numbers for Label 1: {sumLabel1}");

                // Perform majority voting for class 
                var labelCounts = distances.Take(k).GroupBy(x => x.Label).ToDictionary(g => g.Key, g => g.Count());
                string majorityClass = labelCounts.OrderByDescending(x => x.Value).First().Key;
                Console.WriteLine($"Majority Voting results in Class: {majorityClass}");



                // Display corresponding sequence based on predicted class label
                Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();
                sequences.Add("1", new List<double>(new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 8.0, 11.0 }));
                sequences.Add("0", new List<double>(new double[] { 8.0, 1.0, 2.0, 9.0, 10.0, 7.0, 11.0, 15.00, 19.00 }));

                if (sequences.ContainsKey(majorityClass))
                {
                    Console.WriteLine("Corresponding Sequence:");
                    Console.WriteLine(string.Join(", ", sequences[majorityClass]));
                }
                else
                {
                    Console.WriteLine("No corresponding sequence found.");
                }
            }

            Console.WriteLine("KNN ended");
            Console.ReadKey();
        }
        private static void RunMultiSimpleSequenceLearningExperiment()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            sequences.Add("S1", new List<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, }));
            sequences.Add("S2", new List<double>(new double[] { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 }));

            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);
        }

        private static void RunMultiSequenceLearningExperiment()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            //sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 7.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 }));
            //sequences.Add("S2", new List<double>(new double[] { 0.8, 2.0, 0.0, 3.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 2.0, 7.0, 1.0, 9.0, 11.0, 11.0, 10.0, 13.0, 14.0, 11.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0 }));

            sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 8.0, 11.0 }));
            sequences.Add("S2", new List<double>(new double[] { 8.0, 1.0, 2.0, 9.0, 10.0, 7.0, 11.00, 15.00, 19.00 }));

            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);

            //
            // These list are used to see how the prediction works.
            // Predictor is traversing the list element by element. 
            // By providing more elements to the prediction, the predictor delivers more precise result.
            var list1 = new double[] { 1.0, 2.0, 3.0, 4.0, 2.0, 5.0 };
            var list2 = new double[] { 2.0, 3.0, 4.0 };
            var list3 = new double[] { 8.0, 1.0, 2.0 };

        }

        /// <summary>
        /// Loads data from a CSV file.
        /// </summary>
        /// <param name="filePath">Path to the CSV file.</param>
        /// <returns>List of data points.</returns>
        static List<DataPoint> LoadData(string filePath)
        {
            List<DataPoint> data = new List<DataPoint>();

            try
            {
                using (var reader = new StreamReader(filePath))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var values = line.Split(',');

                        double[] features = Array.ConvertAll(values.Take(values.Length - 1).ToArray(), double.Parse);
                        string label = values.Last();

                        data.Add(new DataPoint { Features = features, Label = label });
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred while loading the dataset: {ex.Message}");
                Environment.Exit(1);
            }

            return data;
        }

        /// <summary>
        /// Displays sample data points to the console.
        /// </summary>
        /// <param name="data">List of data points.</param>
        static void DisplaySampleData(List<DataPoint> data)
        {
            Console.WriteLine("Sample Data Points:");
            for (int i = 0; i < Math.Min(5, data.Count); i++) // Display at most 5 data points
            {
                Console.WriteLine($"Data Point {i + 1}: Features [{string.Join(", ", data[i].Features)}], Label: {data[i].Label}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Splits the data into training and testing sets.
        /// </summary>
        /// <param name="data">List of data points.</param>
        /// <param name="trainRatio">Ratio of training data to total data.</param>
        /// <returns>Tuple containing training and testing data sets.</returns>
        static (List<DataPoint>, List<DataPoint>) SplitData(List<DataPoint> data, double trainRatio)
        {
            int trainSize = (int)(data.Count * trainRatio);
            var shuffledData = data.OrderBy(x => Guid.NewGuid()).ToList();
            var trainingData = shuffledData.Take(trainSize).ToList();
            var testingData = shuffledData.Skip(trainSize).ToList();

            return (trainingData, testingData);
        }

        /// <summary>
        /// Prompts the user to enter the value of k (number of neighbors).
        /// </summary>
        /// <returns>The value of k.</returns>
        static int GetKFromUser()
        {
            int k;
            while (true)
            {
                Console.Write("Enter the value of k (number of neighbors): ");
                if (int.TryParse(Console.ReadLine(), out k) && k > 0)
                {
                    return k;
                }
                else
                {
                    Console.WriteLine("Invalid input. Please enter a positive integer.");
                }
            }
        }

        /// <summary>
        /// Performs k-fold cross-validation
        /// </summary>
        /// <param name="data">List of data points.</param>
        /// <param name="k">Value of k (number of neighbors).</param>
        /// <param name="numFolds">Number of folds for cross-validation.</param>
        /// <returns>An array of accuracies for each fold.</returns>
        static double[] CrossValidate(List<DataPoint> data, int k, int numFolds)
        {
            int foldSize = data.Count / numFolds;
            double[] accuracies = new double[numFolds];

            for (int i = 0; i < numFolds; i++)
            {
                // Partition data into training and testing sets
                List<DataPoint> trainingData = data.Take(i * foldSize).Concat(data.Skip((i + 1) * foldSize)).ToList();
                List<DataPoint> testingData = data.Skip(i * foldSize).Take(foldSize).ToList();

                // Train the classifier on training data
                KNNClassifier knn = new KNNClassifier(trainingData);

                // Test the classifier on testing data
                accuracies[i] = TestClassifier(knn, testingData, k);
            }

            return accuracies;
        }

        /// <summary>
        /// Tests the KNN classifier on given testing data.
        /// </summary>
        /// <param name="knn">Instance of the KNN classifier.</param>
        /// <param name="testData">List of testing data points.</param>
        /// <param name="k">Value of k (number of neighbors).</param>
        /// <returns>The accuracy of the classifier on the testing data.</returns>
        static double TestClassifier(KNNClassifier knn, List<DataPoint> testData, int k)
        {
            int correct = 0;

            foreach (var dataPoint in testData)
            {
                string predictedLabel = knn.Predict(dataPoint, k);

                if (predictedLabel == dataPoint.Label)
                    correct++;
            }

            double accuracy = (double)correct / testData.Count * 100;
            return accuracy;
        }


    }
}
