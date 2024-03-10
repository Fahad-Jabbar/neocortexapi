using NeoCortexApi;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeoCortexApiSample
{
    class Program
    {
        /// <summary>
        /// This sample shows a typical experiment code for SP and TM.
        /// You must start this code in debugger to follow the trace.
        /// and TM.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            // SpatialPatternLearning experiment = new SpatialPatternLearning();
            // experiment.Run();

            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            // SequenceLearning experiment = new SequenceLearning();
            // experiment.Run();

            // RunMultiSimpleSequenceLearningExperiment();
            RunMultiSequenceLearningExperiment();
        }

        private static void RunMultiSimpleSequenceLearningExperiment()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            /* sequences.Add("S1", new List<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, }));
             sequences.Add("S2", new List<double>(new double[] { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 }));*/

            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);
        }


        /// <summary>
        /// This example demonstrates how to learn two sequences and how to use the prediction mechanism.
        /// First, two sequences are learned.
        /// Second, three short sequences with three elements each are created und used for prediction. The predictor used by experiment privides to the HTM every element of every predicting sequence.
        /// The predictor tries to predict the next element.
        /// </summary>
        private static void RunMultiSequenceLearningExperiment()
        {
            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 2.0, 2.5, 4.0, 2.8, 5.0 }));
            sequences.Add("S2", new List<double>(new double[] { 8.0, 1.0, 2.0, 9.4, 10.0, 7.0, 11.0 }));
            sequences.Add("S3", new List<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }));
            sequences.Add("S4", new List<double>(new double[] { 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4 }));
            //sequences.Add("S5", new List<double>(new double[] { 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4 }));
            sequences.Add("S6", new List<double>(new double[] { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0 }));
            sequences.Add("S7", new List<double>(new double[] { 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 }));
            //sequences.Add("S8", new List<double>(new double[] { 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0 }));
            //sequences.Add("S9", new List<double>(new double[] { 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 }));

            // sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 2.0, 2.5 , 4.0, 2.8, 5.0, }));
            //sequences.Add("S2", new List<double>(new double[] { 8.0, 1.0, 2.0, 9.4, 10.0, 7.0, 11.00 }));

            //
            // Prototype for building the prediction engine.
            MultiSequenceLearning experiment = new MultiSequenceLearning();
            var predictor = experiment.Run(sequences);

            //
            // These list are used to see how the prediction works.
            // Predictor is traversing the list element by element. 
            // By providing more elements to the prediction, the predictor delivers more precise result.
            var list1 = new double[] { 8.0, 1.0, 2.0 };
            var list2 = new double[] { 7.0, 7.2, 7.4 };
            /*var list3 = new double[] { 4.0, 5.0, 6.0 };*/

            predictor.Reset();
            PredictNextElement(predictor, list1);

            predictor.Reset();
            PredictNextElement(predictor, list2);

            /*predictor.Reset();
            PredictNextElement(predictor, list3);*/
        }

        private static void PredictNextElement(Predictor predictor, double[] list)
        {
            Debug.WriteLine("------------------------------");

            foreach (var item in list)
            {
                var res = predictor.Predict(item);

                if (res.Count > 0)
                {
                    foreach (var pred in res)
                    {
                        Debug.WriteLine($"{pred.PredictedInput} - {pred.Similarity}");
                    }

                    var tokens = res.First().PredictedInput.Split('_');
                    var tokens2 = res.First().PredictedInput.Split('-');
                    Debug.WriteLine($"Predicted Sequence: {tokens[0]}, predicted next element {tokens2.Last()}");
                }
                else
                    Debug.WriteLine("Nothing predicted :(");
            }

            Debug.WriteLine("------------------------------");
        }
    }
}

