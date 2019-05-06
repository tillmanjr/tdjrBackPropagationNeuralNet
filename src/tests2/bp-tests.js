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

// var BackPropagationNeuralNet = require('../neuralNets/backPropagationNeuralNet')
// var trainBackPropagationNeuralNet = require('../neuralNets/trainBackPropagationNeuralNet')
const {
  BackPropagationNeuralNet
 } = require('../neuralNets2')

const {
  createTrainingOptions
} = require('../neuralNets2');

const {
  trainBackPropagationNeuralNet
 } = require('../neuralNets2')

// returns an array of incrementing values
const generateSequenceVector = (elementCount, startAt, incrementBy) => {
  const result = []
  let currValue = startAt
  for (var i = 0; i < elementCount; i++) {
    result.push(currValue)
    currValue += incrementBy
  }
  return result
}

const emitNetworkConfigDetails = (logMessageFn, inputValues, hiddenCount, targetValues) => {
  const inputCount = inputValues.length
  const outputCount = targetValues.length
  const biasCount = hiddenCount + outputCount
  const weightsCount = (inputCount * hiddenCount) + (hiddenCount * outputCount)

  logMessageFn(`\n1. Create ${inputCount}-${hiddenCount}-${outputCount} Neural Network:`)
  logMessageFn(`\t ${inputCount} input nodes\n\t ${hiddenCount} hidden nodes\n\t ${outputCount} output nodes`)
  logMessageFn('\t-------- Requiring ---------')
  logMessageFn(`\t${weightsCount} weight (input * hidden) + (hidden * output )`)
  logMessageFn(`\t ${biasCount} biases (hidden + output )`)
  logMessageFn('\t----------------------------')

  logMessageFn('The fixed input values are:')
  logMessageFn(formatVector(inputValues, 1, 8, true))

  logMessageFn('The fixed target values are:')
  logMessageFn(formatVector(targetValues, 4, 8, true))
}

const emitWeightBiasesDetails = (logMessageFn, initWeights, inputCount, hiddenCount, outputCount) => {
  const weightsCount = (inputCount * hiddenCount) + (hiddenCount * outputCount) + (hiddenCount + outputCount)
  if (weightsCount !=  initWeights.length) {
    logMessageFn('weights count not correct')
    return
  }

  let pos = 0
  let stopAt = inputCount * hiddenCount
  let vals = initWeights.slice(pos, stopAt)
  const inputToHiddenMsg = `Weights: input to hidden\n${formatVector(vals, 3, 4, true)}\n`

  pos = stopAt
  stopAt = hiddenCount + pos
  vals = initWeights.slice(pos, stopAt)
  const hiddenMsg = `Biases: hidden\n${formatVector(vals, 3, 4, true)}\n`

  pos = stopAt
  stopAt = (hiddenCount * outputCount) + pos
  vals = initWeights.slice(pos, stopAt)
  const hiddenToOutputMsg = `Weights: hidden to output\n${formatVector(vals, 3, 2, true)}\n`

  pos = stopAt
  vals = initWeights.slice(pos)
  const outputMsg = `Biases: output\n${formatVector(vals, 3, 2, true)}\n`

  logMessageFn(`${inputToHiddenMsg}${hiddenToOutputMsg}${hiddenMsg}${outputMsg}`)
}

const emitTrainingResults = (logResultsFn, trainingResult, inputValues, hiddenCount, outputCount) => {
  logResultsFn('=================================\n     Training Results\n=================================')
  if (!trainingResult.success) {
    logResultsFn('         FAILED\n')
    logResultsFn(`Training failed in ${trainingResult.epochUsed} epochs.`)
  } else {
    logResultsFn('         SUCCESS\n')
    const finalWeights = trainingResult.neuralNet.getWeights()
    logResultsFn('Final neural network weights and bias values are:')
    emitWeightBiasesDetails(logResultsFn, finalWeights, inputValues.length, hiddenCount, outputCount)

    const outputValues = trainingResult.neuralNet.computeOutputs(inputValues)
    logResultsFn('The output values using final weights are:')
    logResultsFn(formatVector(outputValues, 8, 8, true))

    var finalError = trainingResult.errorMargin// computeError(targetValues, outputValues)
    logResultsFn(`The final error is ${trainingResult.errorMargin}`)
    logResultsFn(`\nEpoch required: ${trainingResult.epochUsed}`)
  }
}

const testBp = (logResultsFn, logMessageFn) => {
  try
  {
    const inputValues = [ 1.0, -2.0, 3.0 ]
    const targetValues = [ 0.1234, 0.8766 ]

    const inputCount = inputValues.length
    const hiddenCount = 4
    const outputCount = targetValues.length
    const weightsCount = (inputCount * hiddenCount) + (hiddenCount * outputCount) + (hiddenCount + outputCount)

    logMessageFn('\nBegin Neural Network training using Back-Propagation test\n')
    emitNetworkConfigDetails(logMessageFn, inputValues, hiddenCount, targetValues)

    const bnn = new BackPropagationNeuralNet(inputCount, hiddenCount, outputCount)

    logMessageFn('\nGenerating arbitrary initial weights and bias values')
    const initWeights = generateSequenceVector(weightsCount, 0.001, 0.001)
    emitWeightBiasesDetails(logMessageFn, initWeights, inputCount, hiddenCount, outputCount)

    logMessageFn('Loading weights and biases into neural network')
    bnn.setWeights(initWeights)

    const options = createTrainingOptions(
      bnn,
      inputValues,
      targetValues,
      logMessageFn
    )

    const trainingResult = trainBackPropagationNeuralNet(options)
    //   bnn,
    //   inputValues,
    //   targetValues,
    //   logMessageFn
    // )

    emitTrainingResults(logResultsFn, trainingResult, inputValues, hiddenCount, outputCount)

    logMessageFn('\nEnd Neural Network Back-Propagation test\n')
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
