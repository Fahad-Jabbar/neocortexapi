using System.Collections.Generic;
using NeoCortexApi.Entities;

namespace NeoCortexApi.Classifiers;

/// <summary>
/// Defines a generic interface for a K-nearest neighbors (KNN) classifier that can learn from and predict input values based on HTM cell activations.
/// </summary>
/// <typeparam name="TIN">The type of the input data.</typeparam>
/// <typeparam name="TOUT">The type of the output data (typically the same as TIN in classification tasks).</typeparam>
public interface IClassifierKnn<TIN, TOUT>
{
    /// <summary>
    /// Teaches the classifier a new input-output association. 
    /// </summary>
    /// <param name="input">The input value to be learned.</param>
    /// <param name="output">An array of cells activated by the input.</param>
    void Learn(TIN input, Cell[] output);

    /// <summary>
    /// Predicts a list of potential input values based on the current state of predictive cells.
    /// </summary>
    /// <param name="predictiveCells">An array of cells whose activation patterns are used to predict the next inputs.</param>
    /// <param name="howMany">Specifies how many top matching inputs to return based on their proximity.</param>
    /// <returns>A list of predicted inputs along with their classification results.</returns>
    List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] predictiveCells, short howMany = 1);
}