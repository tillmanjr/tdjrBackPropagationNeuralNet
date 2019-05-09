const BackPropagationNeuralNet = require ('./backPropagationNeuralNet')
const trainBackPropagationNeuralNet = require ('./trainBackPropagationNeuralNet')
var createTrainingOptions = require('./trainingOptions');

module.exports = {
  BackPropagationNeuralNet,
  trainBackPropagationNeuralNet,
  createTrainingOptions
}