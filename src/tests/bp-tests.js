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

const testBp = () => {
  try
  {
    console.log('\nBegin Neural Network training using Back-Propagation demo\n');
    const xValues = [ 1.0, -2.0, 3.0 ] // Inputs
    let yValues = [] // Outputs
    const tValues = [ 0.1234, 0.8766 ] // Target values

    console.log('The fixed input xValues are:')
    console.log(formatVector(xValues, 1, 8, true))

    console.log('The fixed target tValues are:')
    console.log(formatVector(tValues, 4, 8, true))

    const numInput = 3
    const numHidden = 4
    const numOutput = 2
    const numWeights = (numInput * numHidden) + (numHidden * numOutput) + (numHidden + numOutput)

    const bnn = new BackPropagationNeuralNet(numInput, numHidden, numOutput)

    console.log('\nCreating arbitrary initial weights and bias values')

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

    console.log('\nInitial weights and biases are:')
    console.log(formatVector(initWeights, 3, 8, true))

    console.log('Loading weights and biases into neural network')
    bnn.setWeights(initWeights) 

    const learnRate = 0.5
    const momentum = 0.1
    const maxEpochs = 10000
    const errorThresh = 0.00001
    
    let epoch = 0
    let error = Number.MAX_SAFE_INTEGER
    console.log('\nBeginning training using back-propagation\n')

    while (epoch < maxEpochs) // Train
    {
      yValues = bnn.computeOutputs(xValues)
      error = computeError(tValues, yValues)
      if (error < errorThresh)
      {
        console.log('Found weights and bias values at epoch ' + epoch)
        break;
      }
      bnn.updateWeights(tValues, learnRate, learnRate)
      ++epoch
    } // Train loop

    const finalWeights = bnn.getWeights()
    console.log('Final neural network weights and bias values are:')
    console.log(formatVector(finalWeights, 5, 8, true))

    yValues = bnn.computeOutputs(xValues)
    console.log('\nThe yValues using final weights are:')
    console.log(formatVector(yValues, 8, 8, true))

    var finalError = computeError(tValues, yValues)
    console.log(`The final error is ${finalError}`)

    console.log('End Neural Network Back-Propagation demo\n')
    //Console.ReadLine()
  }
  catch (error)
  {
    console.log('Fatal: ' + error.message)
  }
}

var bpTestsExports = {
  testBp
}

module.exports = bpTestsExports