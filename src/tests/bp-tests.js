"use strict;"

const makeMatrix = require('../libs/matrixUtils')

const {
  cloneVector,
  prefilledVector
} = require('../libs/vectorUtils')

const {
  formatVector,
  formatMatrix
} = require('../formatters/textFormatter')

const {
  computeError
} = require('../libs/mathUtils')

var BackPropagationNeuralNet = require('../backPropagationNeuralNet')
var trainBackPropagationNeuralNet = require('../trainBackPropagationNeuralNet')

const testBp = (logResultsFn, logMessageFn) => {
  try
  {
    logMessageFn('\nBegin Neural Network training using Back-Propagation demo\n');
    const inputValues = [ 1.0, -2.0, 3.0 ] 
    const targetValues = [ 0.1234, 0.8766 ]

    logMessageFn('The fixed input inputValues are:')
    logMessageFn(formatVector(inputValues, 1, 8, true))

    logMessageFn('The fixed target targetValues are:')
    logMessageFn(formatVector(targetValues, 4, 8, true))

    const inputCount = 3
    const hiddenCount = 4
    const outputCount = 2
    const weightsCount = (inputCount * hiddenCount) + (hiddenCount * outputCount) + (hiddenCount + outputCount)

    let bnn = new BackPropagationNeuralNet(inputCount, hiddenCount, outputCount)

    logMessageFn('\nCreating arbitrary initial weights and bias values')

    const initWeights = [
      0.001, 0.002, 0.003, 0.004,
      0.005, 0.006, 0.007, 0.008,
      0.009, 0.010, 0.011, 0.012,

      0.013, 0.014, 0.015, 0.016,

      0.017, 0.018,
      0.019, 0.020,
      0.021, 0.022,
      0.023, 0.024,

      0.025, 0.026
    ]

    logMessageFn('\nInitial weights and biases are:')
    logMessageFn(formatVector(initWeights, 3, 8, true))

    logMessageFn('Loading weights and biases into neural network')
    bnn.setWeights(initWeights) 
    
    bnn = trainBackPropagationNeuralNet(
      bnn,
      inputValues,
      targetValues,
      logMessageFn
    )

    const finalWeights = bnn.getWeights()
    logResultsFn('Final neural network weights and bias values are:')
    logResultsFn(formatVector(finalWeights, 5, 8, true))

    const outputValues = bnn.computeOutputs(inputValues)
    logResultsFn('\nThe outputValues using final weights are:')
    logResultsFn(formatVector(outputValues, 8, 8, true))

    var finalError = computeError(targetValues, outputValues)
    logResultsFn(`The final error is ${finalError}`)

    logMessageFn('End Neural Network Back-Propagation test\n')
  }
  catch (error)
  {
    logMessageFn('Fatal: ' + error.message)
  }
}

var bpTestsExports = {
  testBp
}

module.exports = bpTestsExports