using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApi.Classifiers
{
    public static class EnumExtension
    {
        public static IEnumerable<(T item, int index)> WithIndex<T>(this IEnumerable<T> self)
            => self.Select((item, index) => (item, index));
    }

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

    public class KNeighborsClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
        private DefaultDictionary<string, List<int[]>> _models = new DefaultDictionary<string, List<int[]>>();
        private int _sdrs = 10;

        private int CalculateDistance(int[] sequence1, int[] sequence2)
        {
            // Simple Manhattan distance for simplicity
            return sequence1.Zip(sequence2, (a, b) => Math.Abs(a - b)).Sum();
        }

        private Dictionary<int, int> GetDistanceTable(int[] classifiedSequence, int[] unclassifiedSequence)
        {
            var distanceTable = new Dictionary<int, int>();
            foreach (var index in unclassifiedSequence)
                distanceTable[index] = classifiedSequence.Min(classifiedIndex => Math.Abs(classifiedIndex - index));
            return distanceTable;
        }

        private List<ClassifierResult<string>> Voting(Dictionary<int, List<ClassificationAndDistance>> mapping, int howMany)
        {
            var votes = new DefaultDictionary<string, int>();
            var overLaps = new Dictionary<string, int>();
            var similarity = new Dictionary<string, double>();

            foreach (var key in _models.Keys)
                overLaps[key] = 0;

            foreach (var coordinates in mapping)
            {
                for (int i = 0; i < _models.Count; i++)
                    votes[coordinates.Value[i].Classification] += 1;

                for (int i = 0; i < coordinates.Value.Count; i++)
                {
                    if (coordinates.Value[i].Distance == 0)
                        overLaps[coordinates.Value[i].Classification] += 1;
                }
            }

            var orderedVotes = votes.OrderByDescending(x => x.Value).ToDictionary(x => x.Key, x => x.Value);
            var orderedOverLaps = overLaps.OrderByDescending(x => x.Value).ToDictionary(x => x.Key, x => x.Value);

            foreach (var paired in orderedOverLaps)
                similarity[paired.Key] = paired.Value != 0 ? (double)paired.Value / mapping.Count : 0;

            var result = new List<ClassifierResult<string>>();

            var orderedResults = orderedOverLaps.Values.First() > mapping.Count / 2
                ? orderedOverLaps.Keys
                : orderedVotes.Keys;

            foreach (var key in orderedResults)
            {
                var cls = new ClassifierResult<string>
                {
                    PredictedInput = key,
                    Similarity = similarity[key],
                    NumOfSameBits = overLaps[key]
                };
                result.Add(cls);
            }

            return result.GetRange(0, howMany).ToList();
        }

        public List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] unclassifiedCells, short howMany = 1)
        {
            if (unclassifiedCells.Length == 0)
                return new List<ClassifierResult<TIN>>();

            var unclassifiedSequence = unclassifiedCells.Select(idx => idx.Index).ToArray();
            var mappedElements = new DefaultDictionary<int, List<ClassificationAndDistance>>();

            foreach (var model in _models)
            {
                foreach (var (sequence, idx) in model.Value.WithIndex())
                {
                    foreach (var dict in GetDistanceTable(sequence, unclassifiedSequence))
                        mappedElements[dict.Key].Add(new ClassificationAndDistance(model.Key, dict.Value, idx));
                }
            }

            foreach (var mappings in mappedElements)
                mappings.Value.Sort();

            return Voting(mappedElements, howMany) as List<ClassifierResult<TIN>>;
        }

        public void Learn(TIN input, Cell[] cells)
        {
            var classification = input as string;
            int[] cellIndices = cells.Select(idx => idx.Index).ToArray();

            if (!_models[classification].Exists(seq => seq.SequenceEqual(cellIndices)))
            {
                if (_models[classification].Count > _sdrs)
                    _models[classification].RemoveAt(0);
                _models[classification].Add(cellIndices);
            }
        }

        public void ClearState() => _models.Clear();
    }
}
