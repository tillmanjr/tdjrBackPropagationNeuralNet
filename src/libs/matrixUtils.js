"use strict;"

const {
  prefilledVector
} = require('./vectorUtils')

const makeMatrix = function(rowCount, colCount, prefillValue) {
  const result = []

  for (var i = 0; i <rowCount; i++) {
    result[i] = prefilledVector(colCount, prefillValue)
  }

  return result
}


const matrixUtilsExports = {
  makeMatrix
}

module.exports = makeMatrix