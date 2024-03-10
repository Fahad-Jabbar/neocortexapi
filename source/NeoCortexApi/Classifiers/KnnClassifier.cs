using System;
using System.Collections.Generic;
using NeoCortexApi.Entities;
using System.Linq;
using NeoCortexApi.Classifiers;

namespace NeoCortexApi.Classifiers
{
    // Extension method for IEnumerable to get item and index
    public static class EnumExtension
    {
        public static IEnumerable<(T item, int index)> WithIndex<T>(this IEnumerable<T> self)
            => self.Select((item, index) => (item, index));
    }

    // DefaultDictionary implementation
    public class DefaultDictionary<TKey, TValue> : Dictionary<TKey, TValue> where TValue : new()
    {
        public new TValue this[TKey key]
        {
            get
            {
                if (!TryGetValue(key, out TValue val))
                {
                    val = new TValue();
                    Add(key, val);
                }

                return val;
            }
            set => base[key] = value;
        }
    }

    // ClassificationAndDistance class
    public class ClassificationAndDistance : IComparable<ClassificationAndDistance>
    {
        public string Classification { get; }
        public int Distance { get; }
        public int ClassificationNo { get; }

        public ClassificationAndDistance(string classification, int distance, int classificationNo)
        {
            Classification = classification;
            Distance = distance;
            ClassificationNo = classificationNo;
        }

        public int CompareTo(ClassificationAndDistance other) => Distance.CompareTo(other.Distance);
    }

    // KNearestNeighborClassifier class
    public class KNeighborsClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
        private int _nNeighbors = 10;
        private DefaultDictionary<string, List<List<double>>> _models = new DefaultDictionary<string, List<List<double>>>();
        private int _sdrs = 10;

        // Method to calculate the distance between two points
        private int Distance(List<double> point1, List<double> point2)
        {
            return (int)Math.Sqrt(point1.Select((a, i) => (a - point2[i]) * (a - point2[i])).Sum());
        }

        // Method to get the least distance
        private int LeastValue(ref List<double> classifiedSequence, double unclassifiedPoint)
        {
            int shortestDistance = int.MaxValue;
            int shortestIndex = -1;

            for (int i = 0; i < classifiedSequence.Count; i++)
            {
                int distance = Math.Abs((int)(classifiedSequence[i] - unclassifiedPoint));
                if (distance < shortestDistance)
                {
                    shortestDistance = distance;
                    shortestIndex = i;
                }
            }

            return shortestIndex;
        }

        // Method to calculate distances
        private Dictionary<int, int> GetDistanceTable(List<double> classifiedSequence, ref List<double> unclassifiedSequence)
        {
            var distanceTable = new Dictionary<int, int>();

            foreach (var index in Enumerable.Range(0, unclassifiedSequence.Count))
                distanceTable[index] = LeastValue(ref classifiedSequence, unclassifiedSequence[index]);

            return distanceTable;
        }

        // Method to compute distances and classify
        private List<ClassifierResult<string>> Voting(Dictionary<int, List<ClassificationAndDistance>> mapping, short howMany)
        {
            var votes = new DefaultDictionary<string, int>();
            var overLaps = new Dictionary<string, int>();
            var similarity = new Dictionary<string, double>();

            // Initializing the overlaps with 0
            foreach (var key in _models.Keys)
                overLaps[key] = 0;

            foreach (var coordinates in mapping)
            {
                for (int i = 0; i < _nNeighbors; i++)
                    votes[coordinates.Value[i].Classification] += 1;

                for (int i = 0; i < coordinates.Value.Count; i++)
                {
                    if (coordinates.Value[i].Distance.Equals(0))
                        overLaps[coordinates.Value[i].Classification] += 1;
                }
            }

            var orderedVotes = votes.OrderByDescending(x => x.Value).ToDictionary(x => x.Key, x => x.Value);
            var orderedOverLaps = overLaps.OrderByDescending(x => x.Value).ToDictionary(x => x.Key, x => x.Value);

            foreach (var paired in orderedOverLaps)
            {
                if (paired.Value != 0)
                    similarity[paired.Key] = (double)paired.Value / mapping.Count;
                else
                    similarity[paired.Key] = 0;
            }

            var result = new List<ClassifierResult<string>>();
            // Checks If the sequence have 50% of overlaps if not the data is ordered in voting manner.
            var orderedResults = orderedOverLaps.Values.First() > mapping.Count / 2
                ? orderedOverLaps.Keys
                : orderedVotes.Keys;

            foreach (var key in orderedResults)
            {
                var cls = new ClassifierResult<string>();
                cls.PredictedInput = key;
                cls.Similarity = similarity[key];
                cls.NumOfSameBits = overLaps[key];
                result.Add(cls);
            }

            return result.Count > howMany ? result.GetRange(0, howMany) : result;
        }

        // Method to classify an unclassified sequence
        public List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] unclassifiedCells, short howMany = 1)
        {
            if (unclassifiedCells.Length == 0)
                return new List<ClassifierResult<TIN>>();

            var unclassifiedSequence = unclassifiedCells.Select(x => x.Index).ToList().ConvertAll(i => (double)i); // Convert to List<double>
            var mappedElements = new DefaultDictionary<int, List<ClassificationAndDistance>>();
            _nNeighbors = _models.Values.Count;

            foreach (var model in _models)
            {
                foreach (var (sequence, idx) in model.Value.WithIndex())
                {
                    foreach (var dict in GetDistanceTable(sequence, ref unclassifiedSequence))
                        mappedElements[dict.Key].Add(new ClassificationAndDistance(model.Key, dict.Value, idx));
                }
            }

            foreach (var mappings in mappedElements)
                mappings.Value.Sort(); //Sorting values according to distance

            return Voting(mappedElements, howMany) as List<ClassifierResult<TIN>>;
        }

        private Dictionary<int, int> GetDistanceTable(int[] classifiedSequence, ref List<double> unclassifiedSequence)
        {
            var classifiedSequenceList = classifiedSequence.Select(x => (double)x).ToList();
            var distanceTable = new Dictionary<int, int>();

            foreach (var index in Enumerable.Range(0, unclassifiedSequence.Count))
            {
                int indexInt = (int)index;
                distanceTable[indexInt] = LeastValue(ref classifiedSequenceList, unclassifiedSequence[index]);
            }

            return distanceTable;
        }
        // Method to add new SDRs to the model
        public void Learn(TIN input, Cell[] cells)
        {
            var classification = input as string;
            var cellIndicies = cells.Select(x => (double)x.Index).ToList(); // Convert Index to double

            if (!_models[classification].Exists(seq => seq.SequenceEqual(cellIndicies)))
            {
                if (_models[classification].Count > _sdrs)
                    _models[classification].RemoveAt(0);
                _models[classification].Add(cellIndicies);
            }
        }

        // Method to clear the model
        public void ClearState() => _models.Clear();
    }
}
