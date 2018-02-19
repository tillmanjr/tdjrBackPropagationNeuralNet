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
    let error = Number.MAX_SAFE_INTEGER

    if (logMessageFn) logMessageFn('\nBeginning training using back-propagation\n')

    while (epoch < maxEpochs) // Train
    {
      outputValues = neuralNet.computeOutputs(inputValues)
      error = computeError(targetValues, outputValues)
      if (error < errorThreshold)
      {
        if (logMessageFn) logMessageFn('Found weights and bias values at epoch ' + epoch)
        break;
      }
      neuralNet.updateWeights(targetValues, learnRate, learnRate)
      ++epoch
    }

    return neuralNet
}

module.exports = trainBackPropagationNeuralNet;