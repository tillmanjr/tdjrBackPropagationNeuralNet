"use strict;"

const BackPropagationNeuralNet = require('./backPropagationNeuralNet')
const {
  computeError
 } = require('./libs/mathUtils')

const trainBackPropagationNeuralNet = (
    neuralNet,
    inputValues,
    targetValues,
    logMessageFn = null,
    initialWeights = null, // if null assumes net contains initial weights, otherwise applies these
    errorThreshold = 0.00001,
    learnRate = 0.5,
    momentum = 0.1,
    maxEpochs = 1000
  ) => {
    if (initialWeights) neuralNet.setWeights(initialWeights)

    let outputValues = []
    
    let epoch = 0
    let errorMargin = Number.MAX_SAFE_INTEGER
    let success = false

    if (logMessageFn) logMessageFn('\nBeginning training using back-propagation\n')

    while (epoch < maxEpochs) // Train
    {
      outputValues = neuralNet.computeOutputs(inputValues)
      errorMargin = computeError(targetValues, outputValues)
      if (errorMargin < errorThreshold)
      {
        if (logMessageFn) logMessageFn('Found weights and bias values at epoch ' + epoch)
        success = true
        break;
      }
      neuralNet.updateWeights(targetValues, learnRate, learnRate)
      ++epoch
    }

    return {
      neuralNet,
      success,
      epochUsed: epoch,
      errorMargin
    }
}

module.exports = trainBackPropagationNeuralNet;