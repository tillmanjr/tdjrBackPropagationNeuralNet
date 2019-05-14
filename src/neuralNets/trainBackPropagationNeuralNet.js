"use strict;"

const BackPropagationNeuralNet = require('./backPropagationNeuralNet')
const {
  computeError
 } = require('../libs/mathUtils')


const trainBackPropagationNeuralNet = (options) => {
    // yeah, yeah, destructure the following option values
    const neuralNet = options.neuralNet
    const inputValues = options.inputValues
    const targetValues = options.targetValues
    const logMessageFn = options.logMessageFn
    const initialWeights = options.initialWeights
    const errorThreshold = options.errorThreshold
    const learnRate = options.learnRate
    // const momentum = options.momentum
    const maxEpochs = options.maxEpochs

    const log = (message) => {
      if (logMessageFn) logMessageFn(message)
    }

    if (initialWeights) neuralNet.setWeights(initialWeights)

    let outputValues = []

    let epoch = 0
    let errorMargin = Number.MAX_SAFE_INTEGER
    let success = false

    log('\nBeginning training using back-propagation\n')

    while (epoch < maxEpochs) // Train
    {
      outputValues = neuralNet.computeOutputs(inputValues)
      errorMargin = computeError(targetValues, outputValues)
      if (errorMargin < errorThreshold)
      {
        log('Found weights and bias values at epoch ' + epoch)
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

module.exports = trainBackPropagationNeuralNet