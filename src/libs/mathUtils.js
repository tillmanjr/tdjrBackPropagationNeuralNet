"use strict;"

const createBounds = require('./bounds').createBounds

// both hyperbolic tangent and the logistic function go constant
// outside the range bounded by -45 and 45 degress
const bounds = createBounds(-45.0, 45.0)

// is x less than -45.0
const isBelowBounds = x => bounds.isBelowBounds(x)

// is value greaaterThan 45.0
const isAboveBounds = x => bounds.isAboveBounds(x)

// Computes a Sigmoid value for X using the logistic function
const sigmoid = function (x) {
  return isBelowBounds(x)
        ? 0.0
        : isAboveBounds(x)
          ? 1.0
          : 1.0 / (1.0 + Math.exp(-x))
}
// Computes the HyperbolicTangent of X (in degrees)
const hyperTan = (x) => {
  return isBelowBounds(x)
        ? -1.0
        : isAboveBounds(x)
          ? 1.0
          : Math.tanh(x)
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
