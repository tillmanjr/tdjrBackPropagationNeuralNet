"use strict;"

// Computes the Sigmoid value of X
const sigmoid = function (x) {
  if (x < -45.0) return 0.0
  if (x > 45.0) return 1.0
  return 1.0 / (1.0 + Math.exp(-x))
}

// Computes the HyperTan of X
const hyperTan = function (x) {
  if (x < -45.0) return -1.0;
  if (x > 45.0) return 1.0;
  return Math.tanh(x);
}

// derivative of Tanh = (1 + y) * (1 - y)
const derivativeTanH = (y) => (1 + y) * (1 - y)

// derivative of log-sigmoid is y(1-y)
const derivativeLogSigmoid = (y) => (1 - y) * y

const computeError = (
  tValues,
  yValues
) => {
  let sum = 0.0;
  for (var i = 0; i < tValues.length; ++i)
    sum += (tValues[i] - yValues[i]) * (tValues[i] - yValues[i]);
  return Math.sqrt(sum);
}

module.exports = {
  computeError,
  derivativeLogSigmoid,
  derivativeTanH,
  hyperTan,
  sigmoid
}
