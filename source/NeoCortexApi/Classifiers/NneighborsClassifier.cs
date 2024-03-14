
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApi.Classifiers
{
    internal class NneighborsClassifier
    {
    }
}



/*
Introducing the K-Nearest Neighbor (KNN) Classifier:

The KNN (K-Nearest-Neighbor) Classifier, seamlessly integrated with the Neocortex API, stands as a beacon of simplicity and effectiveness in the realm of machine learning.
It offers a straightforward approach to classification, relying on the proximity of data points in a feature space to make predictions.

Here's how it works:

Training the Model:
The KNN Classifier begins by ingesting a sequence of values along with their preassigned labels. This dataset is used to train the model, resulting in a dictionary mapping 
labels to their respective sequences.

Example:

_models = {
    "A": [[1, 3, 4, 7, 12, 13, 14], [2, 3, 5, 6, 7, 8, 12]],
    "B": [[0, 4, 5, 6, 9, 10, 13], [2, 3, 4, 5, 6, 7, 8]],
    "C": [[1, 4, 5, 6, 8, 10, 15], [1, 2, 7, 8, 13, 15, 16]]
}
Making Predictions:
Once trained, the KNN Classifier is ready to label unclassified sequences. When presented with a new sequence, it identifies the k nearest neighbors from the training dataset.
By considering the majority class among these neighbors, the classifier predicts the label for the unclassified sequence.

Example:

unknown = [1, 3, 4, 7, 12, 14, 15]
The Verdict:
The output of the KNN Classifier is a list of ClassifierResult objects, sorted in descending order of match closeness. 
The closest match is labeled "A," followed by "B," and so forth.

In summary, the KNN Classifier exemplifies the power of simplicity in machine learning, providing accurate predictions based on the proximity of data points in the feature space.
*/