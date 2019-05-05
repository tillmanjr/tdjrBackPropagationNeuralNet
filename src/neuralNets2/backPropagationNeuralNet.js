"use strict;"

const makeMatrix = require('../libs/matrixUtils')

const {
  derivativeLogSigmoid,
  derivativeTanH,
  sigmoid,
  hyperTan
} = require('../libs/mathUtils')

const {
  cloneVector,
  prefilledVector
} = require('../libs/vectorUtils')

const {
  InvalidModelException
} = require('../libs/exceptions')

const DEFAULT_VECTOR_PREFILL_VALUE = 0.0

let pipe = (...fns) => x => fns.reduce((v, f) => f(v), x)

class BackPropagationNeuralNet {

  // TODO: extract computed properties into _init function
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

  // calculates and returns the WeightsCount taking into account all dimmensionalities
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

    k = this._weightInputHiddenWeights(k, weights)

    k = this._weightHiddenBiases(k, weights)

    k = this._weightHiddenOutputWeights(k, weights)

    this._weightHiddenOutputs(k, weights)
  }

  getWeights() {
    const weightsCount = this.getWeightsCount()

    let result = prefilledVector(weightsCount, 0.0)
    let k = 0;

    let carrier = this.createCarrier(result, k)
    carrier = this._reCalculateHiddenWeights( carrier )
    carrier = this._recalculateHiddenBiases(carrier)
    carrier = this._recalculateHiddenOutoutWeights(carrier)
    carrier = this._recalculateOutputBiases(carrier)

    return carrier.array
  }

  getOutputs() {
    return cloneVector(this.outputs)
  }

  /* #region  computeOutputs helper function: matrix and array functions */
  // TODO: convert this set of methods to functions

  setArraySubrangeTo( array, start, count, value) {
    const end = start + count
    for(let i = start; i < end; i++) {
      array[i] = value
    }
    return array
  }

  //
  copyValuesFromArray( copyTo, copyFrom, transformFn ) {
    return this.copyValuesFromSubArray( copyTo, copyFrom, copyFrom.length, transformFn )
  }

  copyValuesFromSubArray( copyTo, copyFrom, copyCount, transformFn ) {
    if (copyCount > copyTo.length)
      throw new Exception(`copyFromArray Exception. copyFrom length ${copyFrom.length} exceeds copyTo length ${copyTo.length}`);

    const identityFn = (x) => x
    transformFn = transformFn ? transformFn : identityFn

    for (let i = 0; i < copyCount; ++i) {
      copyTo[i] = transformFn(copyFrom[i])
    }
    return copyTo
  }


  addToValuesFromArray( copyTo, copyFrom ) {
    if (copyFrom.length > copyTo.length)
      throw new Exception(`copyFromArray Exception. copyFrom length ${copyFrom.length} exceeds copyTo length ${copyTo.length}`);

    for (let i = 0; i < copyFrom.length; ++i) {
      copyTo[i] += copyFrom[i]
    }
    return copyTo
  }

  // for each 0 >> copyCountDim1 as dim1
  //  do each 0 >> copyCountDim0 as dim0
  //    value copyTo[dim1] += copyFromArray[dim0] * copyFromMatrix[dim0][dim1]
  addToValuesFromArrayMatrixProduct(
    copyTo,         // array being modified
    copyFromArray,  // array: values are copied from for processing
    copyFromMatrix, // matrix: values are copied from for processing
    copyCountDim0,  // = copyFromMatrix dimension 0 limit AND <= copyFromArray.length
    copyCountDim1   // = copyFromMatrix dimension 1 limit AND <= copyTo.length
  ) {
    for (var j = 0; j < copyCountDim1; ++j)
      for (var i = 0; i < copyCountDim0; ++i)
        copyTo[j] += copyFromArray[i] * copyFromMatrix[i][j]
  }

  /* #endregion */

  computeOutputs(
    inputValues
  ) {
    if (inputValues.length != this.inputCount)
      throw new InvalidModelException(`Inputs array length ${this.inputs.length} does not match NN inputCount value ${this.inputCount}`);

    this.setArraySubrangeTo( this.outputSums, 0, this.hiddenCount, 0.0 )

     // copy x-values to inputs
    this.copyValuesFromArray( this.inputs, inputValues)

     // compute hidden layer weighted sums
    this.addToValuesFromArrayMatrixProduct(
      this.hiddenSums,
      this.inputs,
      this.inputHiddenWeights,
      this.inputCount,
      this.hiddenCount )

    // add biases to hidden sums
    this.addToValuesFromArray(this.hiddenSums, this.hiddenBiases)

    // apply tanh activation
    this.copyValuesFromArray( this.hiddenOutputs, this.hiddenSums, hyperTan )

    // compute output layer weighted sums
    this.addToValuesFromArrayMatrixProduct(
      this.outputSums,
      this.hiddenOutputs,
      this.hiddenOutputWeights,
      this.hiddenCount,
      this.outputCount )

    // add biases to output sums
    this.addToValuesFromArray(this.outputSums, this.outputBiases)

    // apply log-sigmoid activation
    this.copyValuesFromSubArray( this.outputs, this.outputSums, this.outputCount, sigmoid )

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
    this._computeOutputGradients(targetValues)

    // 2. compute hidden gradients. assumes tanh!
    this._computeHiddenGradients()

    // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order)
    this._updateInputsToHiddenWeights (learn, momentum)

    // 4. update hidden biases
    this._updateHiddenBiases(learn, momentum)

    // 5. update hidden to output weights
    this._updateHiddenToOutputWeights (learn, momentum)

    // 6. update hidden to output biases
    this._updateHiddenToOutputBiases(learn, momentum)
  }

/* #region Computation helper methods */

/* #region  setWeights helper functions */

_weightInputHiddenWeights (k, weights) {
  for (var i = 0; i < this.inputCount; ++i)
    for (var j = 0; j < this.hiddenCount; ++j)
      this.inputHiddenWeights[i][j] = weights[k++]
  return k
}

_weightHiddenBiases (k, weights) {
  for (var i = 0; i < this.hiddenCount; ++i)
    this.hiddenBiases[i] = weights[k++]
  return k
}

_weightHiddenOutputWeights (k, weights) {
  for (var i = 0; i < this.hiddenCount; ++i)
    for (var j = 0; j < this.outputCount; ++j)
      this.hiddenOutputWeights[i][j] = weights[k++]
  return k
}

_weightHiddenOutputs (k, weights) {
  for (var i = 0; i < this.outputCount; ++i)
    this.outputBiases[i] = weights[k++]
  return k
}

/* #endregion */

/* #region  getWeights helper functions */

  // simple wrapper for an array and an offset
  createCarrier(
    array,
    offset
  ) {
    return {
      array,
      offset
    }
  }

  _reCalculateHiddenWeights( carrier ) {
    let k = carrier.offset
    for (var i = 0; i < this.inputHiddenWeights.length; ++i) {
      for (var j = 0; j < this.inputHiddenWeights[0].length; ++j) {
        carrier.array[k++] = this.inputHiddenWeights[i][j]
      }
    }
    carrier.offset = k
    return carrier
  }
  _recalculateHiddenBiases( carrier) {
    let k = carrier.offset
      for (var i = 0; i < this.hiddenBiases.length; ++i) {
        carrier.array[k++] = this.hiddenBiases[i]
      }
    carrier.offset = k
    return carrier
  }
  _recalculateHiddenOutoutWeights( carrier ){
    let k = carrier.offset
    for (var i = 0; i < this.hiddenOutputWeights.length; ++i) {
      for (var j = 0; j < this.hiddenOutputWeights[0].length; ++j) {
        carrier.array[k++] = this.hiddenOutputWeights[i][j]
      }
    }
    carrier.offset = k
    return carrier
  }

  _recalculateOutputBiases( carrier ){
    let k = carrier.offset
    for (var i = 0; i < this.outputBiases.length; ++i) {
      carrier.array[k++] = this.outputBiases[i]
    }
    carrier.offset = k
    return carrier
  }

  /* #endregion */

/* #region  updateWeights calculation helper funtions */

  // compute output gradients. assumes log-sigmoid!
  _computeOutputGradients(targetValues) {
    for (var i = 0; i < this.outputGradients.length; ++i) {
      const derivative = derivativeLogSigmoid(this.outputs[i])
      this.outputGradients[i] = derivative * (targetValues[i] - this.outputs[i]) // oGrad = (1 - O)(O) * (T-O)
    }
  }

  // 2. compute hidden gradients. assumes tanh!
  _computeHiddenGradients() {
    for (var i = 0; i < this.hiddenGradients.length; ++i) {
      const derivative = derivativeTanH(this.hiddenOutputs[i])
      let sum = 0.0;
      for (var j = 0; j < this.outputCount; ++j) // each hidden delta is the sum of outputCount terms
        sum += this.outputGradients[j] * this.hiddenOutputWeights[i][j]; // each downstream gradient * outgoing weight
      this.hiddenGradients[i] = derivative * sum; // hiddenGradient = (1-O)(1+O) * E(outputGradients*oWts)
    }
  }

  _updateInputsToHiddenWeights (learn, momentum) {
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
  }

  _updateHiddenBiases (learn, momentum) {
    for (var i = 0; i < this.hiddenBiases.length; ++i)
    {
      const delta = learn * this.hiddenGradients[i] * 1.0 // the 1.0 is the constant input for any bias; could leave out
      this.hiddenBiases[i] += delta
      this.hiddenBiases[i] += momentum * this.hiddenPrevBiasesDelta[i]
      this.hiddenPrevBiasesDelta[i] = delta // save delta
    }
  }

  _updateHiddenToOutputWeights (learn, momentum) {
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
  }

  _updateHiddenToOutputBiases (learn, momentum) {
    for (var i = 0; i < this.outputBiases.Length; ++i)
    {
      const delta = learn * this.outputGradients[i] * 1.0
      this.outputBiases[i] += delta
      this.outputBiases[i] += momentum * this.outputPrevBiasesDelta[i]
      this.outputPrevBiasesDelta[i] = delta
    }
  }

/* #endregion */

/* #endregion */

}

module.exports = BackPropagationNeuralNet;