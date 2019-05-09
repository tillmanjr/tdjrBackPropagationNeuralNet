"use strict;"

const {
  sigmoid,
  hyperTan
} = require('../libs/mathUtils')

const {
  formatVector,
  formatMatrix
} = require('../formatters/textFormatter')

const {
  rotateVector
} = require('../libs/vectorUtils')

/* Formatting constants */
const TEST_SEPARATOR = '----------------------------'
const NEW_LINE = '\n'

const RUN_ALL_HEADER_SEPARARATOR = '******************************************************************'
const RUN_ALL_BEGIN_MESSAGE = 'Runs all tests. Tests emit values to stdout for visual comparison.'
const RUN_ALL_COMPLETED_MESSAGE = 'All tests completed'

const MATHUTILS_DEFAULT_X = 1.0
const FIXED_DECIMALS_DEFAULT = 2
const VALUES_PER_LINE_DEFAULT = 3
const APPEND_BLANK_LINE_DEFAULT = true

const emitTextValue = (text) => console.log(text)
const emitEmptyLine = () => console.log()

const emitTestHeader = (message) => {
  emitTextValue(TEST_SEPARATOR)
  emitTextValue(message)
  emitEmptyLine()
} 

const emitRunAllHeader = () => {
  emitTextValue(RUN_ALL_HEADER_SEPARARATOR)
  emitTextValue(RUN_ALL_BEGIN_MESSAGE)
  emitTextValue(RUN_ALL_HEADER_SEPARARATOR)
}


const getSampleVector = () => 
  [
    123.456,
    234.567,
    345.678,
    456.789,
    567.890,
    678.901,
    789.012,
    890.123,
    901.234,
    123.456,
    234.567,
    345.678,
    456.789,
    567.890,
    678.901,
    789.012,
    890.123,
    901.234,
  ]

const getSampleMatrix = () => {
  const result = []

  for (var i = 0; i <4; i++) {
    result[i] = []
  }

  const sampleVector = getSampleVector()

  for (var i = 0; i <4; i++) {
    for (var j = 0; j <4; j++) {
      result[i][j] = sampleVector[i*4 + j]
    }
  }

  return result
}

const testRotateVector = () => {
  emitTestHeader('Begin rotateVector tests')

  const sampleVector = getSampleVector()
  const fixedDec = FIXED_DECIMALS_DEFAULT
  const valuesPerLine = sampleVector.length
  const appendBlankLine = false

  let result = '';
  result += ` 0\t${formatVector(rotateVector(sampleVector, 0), fixedDec, valuesPerLine, appendBlankLine)}\n`
  result += ` 1\t${formatVector(rotateVector(sampleVector, 1), fixedDec, valuesPerLine, appendBlankLine)}\n`
  result += ` 2\t${formatVector(rotateVector(sampleVector, 2), fixedDec, valuesPerLine, appendBlankLine)}\n`
  result += ` 3\t${formatVector(rotateVector(sampleVector, 3), fixedDec, valuesPerLine, appendBlankLine)}\n`

  result += `-1\t${formatVector(rotateVector(sampleVector, -1), fixedDec, valuesPerLine, appendBlankLine)}\n`
  result += `-2\t${formatVector(rotateVector(sampleVector, -2), fixedDec, valuesPerLine, appendBlankLine)}\n`
  result += `-3\t${formatVector(rotateVector(sampleVector, -3), fixedDec, valuesPerLine, appendBlankLine)}\n`

  emitTextValue(result)
}

const mathUtilsTests = () => {
  emitTestHeader('Begin mathUtils tests')

  const xIn = MATHUTILS_DEFAULT_X

  const sigmoidResult = sigmoid(xIn)
  emitTextValue(`Sigmoid of ${xIn} = ${sigmoidResult}`)

  const hyperTanResult = hyperTan(xIn)
  emitTextValue(`HyperTan of ${xIn} = ${hyperTanResult}`)
}

const formattedVectorTests = () => {
  emitTestHeader('Begin formattedVector tests')

  const sampleVector = getSampleVector()

  const fixedDec = FIXED_DECIMALS_DEFAULT
  const valuesPerLine = VALUES_PER_LINE_DEFAULT
  const appendBlankLine = APPEND_BLANK_LINE_DEFAULT

  const formattedVector = formatVector(sampleVector, fixedDec, valuesPerLine, appendBlankLine)
  
  emitTextValue(formattedVector)
}

const formattedMatrixTests = () => {
  emitTestHeader('Begin formattedMatrix tests')

  const sampleMatrix = getSampleMatrix()
  const formattedMatrix = formatMatrix(sampleMatrix, 4, 2)
  
  emitTextValue(formattedMatrix)
}

const runAll = () => {
  emitRunAllHeader()

  mathUtilsTests()
  formattedVectorTests()
  testRotateVector()
  formattedMatrixTests()

  emitTextValue(RUN_ALL_COMPLETED_MESSAGE)
} 

const formatTests = {
  runAll,
  mathUtilsTests,
  formattedVectorTests,
  formattedMatrixTests
}

module.exports = formatTests