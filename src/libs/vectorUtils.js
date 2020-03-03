"use strict;"

const generateSequence = require('./sequenceGenerator').generateSequence
const createRange = require('./arrayRange').createRange

const prefilledVector = (len, prefillValue) => {
  const seq = generateSequence(0, len, 1)
  return [...seq].map( _ => prefillValue)
}

const cloneVector = (vector) => vector.map( (x) => x)

const normalizeVectorRotateBy = (vectorLen, rotateBy) => {
  if (vectorLen == rotateBy) return 0
  if (rotateBy == 0) return 0

  if (rotateBy > 0) {
    return rotateBy < vectorLen ? rotateBy : vectorLen % rotateBy
  }

  return vectorLen + rotateBy
}

const rotateVector = (vector, rotateBy) => {
  const vectorLen = vector.length
  const _rotateBy =  normalizeVectorRotateBy(vectorLen, rotateBy);

  if (_rotateBy === 0) return vector

  const vectorXRange = createRange(0, vectorLen - _rotateBy - 1, 1)
  const vX = vectorXRange.map( x => vector[x + _rotateBy])

  const vectorYRange = createRange(0, _rotateBy - 1, 1)
  const vY = vectorYRange.map( y => vector[y] )

  return vX.concat(vY)

  // old code
  // const result = []
  // for (let x = 0; x < vectorLen - _rotateBy; x++){
  //   result.push( vector[x + _rotateBy])
  // }

  // for (let y = 0; y < _rotateBy; y++) {
  //  result.push(vector[y])
  // }

  // return result
}

const exportUtils = {
  rotateVector,
  cloneVector,
  prefilledVector
}

module.exports = exportUtils