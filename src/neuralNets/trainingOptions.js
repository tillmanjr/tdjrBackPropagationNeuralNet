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


class testTarget {
  constructor (...options) {

    console.log('testTarget : ', testTarget)
  }

}

const testNet = {objIs: 'test neural net'}
const testIputValues = {objIs: 'test inputValues'}
const testTargetValues = {objIs: 'test targetValues'}
const testlogMessageFn = {objIs: 'test logMessageFn'}
const testOptions = createTrainingOptions(
  testNet,
  testIputValues,
  testTargetValues,
  testlogMessageFn
)
const foo = new testTarget(testOptions)
console.log('foo : ', foo)

module.exports = createTrainingOptions