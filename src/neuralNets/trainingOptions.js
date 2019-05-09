const createTrainingOptions = (
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
  return {
    neuralNet,
    inputValues,
    targetValues,
    logMessageFn,
    initialWeights, // if null assumes net contains initial weights, otherwise applies these
    errorThreshold,
    learnRate,
    momentum,
    maxEpochs
  }
}

module.exports = createTrainingOptions