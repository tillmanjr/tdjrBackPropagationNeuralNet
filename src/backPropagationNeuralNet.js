"use strict;"

const makeMatrix = require('./libs/matrixUtils')

const {
  derivativeLogSigmoid,
  derivativeTanH,
  sigmoid,
  hyperTan
} = require('./libs/mathUtils')

const {
  cloneVector,
  prefilledVector
} = require('./libs/vectorUtils')

const {
  InvalidModelException
} = require('./libs/exceptions')

const DEFAULT_VECTOR_PREFILL_VALUE = 0.0


class BackPropagationNeuralNet {

  constructor(inputCount, hiddenCount, outputCount) {
    this.inputCount = inputCount
    this.hiddenCount = hiddenCount
    this.outputCount = outputCount

    this.inputs = prefilledVector(inputCount, DEFAULT_VECTOR_PREFILL_VALUE)

    this.inputHiddenWeights = makeMatrix(inputCount, hiddenCount, 0.0)
    
    this.hiddenBiases = prefilledVector(hiddenCount, DEFAULT_VECTOR_PREFILL_VALUE)
    this.hiddenSums = prefilledVector(hiddenCount, DEFAULT_VECTOR_PREFILL_VALUE)
    this.hiddenOutputs = prefilledVector(hiddenCount, DEFAULT_VECTOR_PREFILL_VALUE)

    this.hiddenOutputWeights = makeMatrix(hiddenCount, outputCount, 0.0)
    this.outputBiases = prefilledVector(outputCount, DEFAULT_VECTOR_PREFILL_VALUE)
    this.outputSums = prefilledVector(outputCount, DEFAULT_VECTOR_PREFILL_VALUE)
    this.outputs = prefilledVector(outputCount, DEFAULT_VECTOR_PREFILL_VALUE)

    this.outputGradients = prefilledVector(outputCount, DEFAULT_VECTOR_PREFILL_VALUE) // output gradients for back-propagation
    this.hiddenGradients = prefilledVector(hiddenCount, DEFAULT_VECTOR_PREFILL_VALUE) // hidden gradients for back-propagation

    this.inputHiddenPrevWeightsDelta = makeMatrix(inputCount, hiddenCount, 0.0)
    this.hiddenPrevBiasesDelta = prefilledVector(hiddenCount, DEFAULT_VECTOR_PREFILL_VALUE)
    this.hiddenOutputPrevWeightsDelta = makeMatrix(hiddenCount, outputCount, 0.0)
    this.outputPrevBiasesDelta = prefilledVector(outputCount, DEFAULT_VECTOR_PREFILL_VALUE)
  }

  getWeightsCount() {
    // weights[] is ordered: input-to-hidden wts, hidden biases, hidden-to-output wts, output biases
    return  ( this.inputCount * this.hiddenCount ) +
            ( this.hiddenCount * this.outputCount ) +
              this.hiddenCount +
              this.outputCount;
  }

  checkWeightsCount( weights ) {
    return this.getWeightsCount() === weights.length;
  }

  setWeights(
    weights // assumes weights[] has order of: input-to-hidden wts, hidden biases, hidden-to-output wts, output biases
  ) { 
    if (!this.checkWeightsCount(weights)) {      
      throw new InvalidModelException(`The weights array length: ${weights.length} 
        does not match the total number of weights and biases`)
    }

    let k = 0; // pointer into weights param

    for (var i = 0; i < this.inputCount; ++i)
      for (var j = 0; j < this.hiddenCount; ++j)
        this.inputHiddenWeights[i][j] = weights[k++]

    for (var i = 0; i < this.hiddenCount; ++i)
      this.hiddenBiases[i] = weights[k++]

    for (var i = 0; i < this.hiddenCount; ++i)
      for (var j = 0; j < this.outputCount; ++j)
        this.hiddenOutputWeights[i][j] = weights[k++]

    for (var i = 0; i < this.outputCount; ++i)
      this.outputBiases[i] = weights[k++]
  }

  getWeights() {
    const weightsCount = this.getWeightsCount()
    
    const result = prefilledVector(weightsCount, 0.0)

    let k = 0;

    for (var i = 0; i < this.inputHiddenWeights.length; ++i) {
      for (var j = 0; j < this.inputHiddenWeights[0].length; ++j) {
        result[k++] = this.inputHiddenWeights[i][j]
      }
    }

    for (var i = 0; i < this.hiddenBiases.length; ++i) {
      result[k++] = this.hiddenBiases[i]
    }

    for (var i = 0; i < this.hiddenOutputWeights.length; ++i) {
      for (var j = 0; j < this.hiddenOutputWeights[0].length; ++j) {
        result[k++] = this.hiddenOutputWeights[i][j]
      }
    }

    for (var i = 0; i < this.outputBiases.length; ++i) {
      result[k++] = this.outputBiases[i]
    }

    return result
  }

  getOutputs() {
    return cloneVector(this.outputs)
  }

  computeOutputs(
    inputValues
  ) {
    if (inputValues.length != this.inputCount)
      throw new InvalidModelException(`Inputs array length ${this.inputs.length} does not match NN inputCount value ${this.inputCount}`);

    for (var i = 0; i < this.hiddenCount; ++i)
      this.hiddenSums[i] = 0.0

    for (var i = 0; i < this.outputCount; ++i)
      this.outputSums[i] = 0.0

    for (var i = 0; i < inputValues.length; ++i) // copy x-values to inputs
      this.inputs[i] = inputValues[i]

    for (var j = 0; j < this.hiddenCount; ++j)  // compute hidden layer weighted sums
      for (var i = 0; i < this.inputCount; ++i)
        this.hiddenSums[j] += this.inputs[i] * this.inputHiddenWeights[i][j]

    for (var i = 0; i < this.hiddenCount; ++i)  // add biases to hidden sums
      this.hiddenSums[i] += this.hiddenBiases[i]

    for (var i = 0; i < this.hiddenCount; ++i)   // apply tanh activation
      this.hiddenOutputs[i] = hyperTan(this.hiddenSums[i])

    for (var j = 0; j < this.outputCount; ++j)   // compute output layer weighted sums
      for (var i = 0; i < this.hiddenCount; ++i)
        this.outputSums[j] += this.hiddenOutputs[i] * this.hiddenOutputWeights[i][j]

    for (var i = 0; i < this.outputCount; ++i)  // add biases to output sums
      this.outputSums[i] += this.outputBiases[i]

    for (var i = 0; i < this.outputCount; ++i)   // apply log-sigmoid activation
      this.outputs[i] = sigmoid(this.outputSums[i])

    const result = cloneVector(this.outputs)
    return result;
  } 

  updateWeights (
    targetValues,
    learn,
    momentum
  ) {
    // assumes that SetWeights and ComputeOutputs have been called and so inputs and outputs have values
    if (targetValues.length != this.outputCount)
      throw new InvalidModelException('Target values not same Length as output in updateWeights');

    // 1. compute output gradients. assumes log-sigmoid!
    for (var i = 0; i < this.outputGradients.length; ++i)
    {
      const derivative = derivativeLogSigmoid(this.outputs[i])
      this.outputGradients[i] = derivative * (targetValues[i] - this.outputs[i]) // oGrad = (1 - O)(O) * (T-O)
    }

    // 2. compute hidden gradients. assumes tanh!
    for (var i = 0; i < this.hiddenGradients.length; ++i)
    {
      const derivative = derivativeTanH(this.hiddenOutputs[i])
      let sum = 0.0;
      for (var j = 0; j < this.outputCount; ++j) // each hidden delta is the sum of outputCount terms
        sum += this.outputGradients[j] * this.hiddenOutputWeights[i][j]; // each downstream gradient * outgoing weight
      this.hiddenGradients[i] = derivative * sum; // hiddenGradient = (1-O)(1+O) * E(outputGradients*oWts)
    }

    // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order)
    for (var i = 0; i < this.inputHiddenWeights.length; ++i) // 0..2 (3)
    {
      for (var j = 0; j < this.inputHiddenWeights[0].length; ++j) // 0..3 (4)
      {
        const delta = learn * this.hiddenGradients[j] * this.inputs[i]; // compute the new delta = "eta * hGrad * input"
        this.inputHiddenWeights[i][j] += delta // update
        this.inputHiddenWeights[i][j] += momentum * this.inputHiddenPrevWeightsDelta[i][j] // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
        this.inputHiddenPrevWeightsDelta[i][j] = delta // save the delta for next time
      }
    }

    // 4. update hidden biases
    for (var i = 0; i < this.hiddenBiases.length; ++i)
    {
      const delta = learn * this.hiddenGradients[i] * 1.0 // the 1.0 is the constant input for any bias; could leave out
      this.hiddenBiases[i] += delta
      this.hiddenBiases[i] += momentum * this.hiddenPrevBiasesDelta[i]
      this.hiddenPrevBiasesDelta[i] = delta // save delta
    }

    // 5. update hidden to output weights
    for (var i = 0; i < this.hiddenOutputWeights.length; ++i)  // 0..3 (4)
    {
      for (var j = 0; j < this.hiddenOutputWeights[0].length; ++j) // 0..1 (2)
      {
        const delta = learn * this.outputGradients[j] * this.hiddenOutputs[i]  // hiddenOutputs are inputs to next layer
        this.hiddenOutputWeights[i][j] += delta
        this.hiddenOutputWeights[i][j] += momentum * this.hiddenOutputPrevWeightsDelta[i][j]
        this.hiddenOutputPrevWeightsDelta[i][j] = delta
      }
    }

    // 6. update hidden to output biases
    for (var i = 0; i < this.outputBiases.Length; ++i)
    {
      const delta = learn * this.outputGradients[i] * 1.0
      this.outputBiases[i] += delta
      this.outputBiases[i] += momentum * this.outputPrevBiasesDelta[i]
      this.outputPrevBiasesDelta[i] = delta
    }
  }

}

module.exports = BackPropagationNeuralNet;