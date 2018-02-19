"use strict;"

const {
  runAll,
  mathUtilsTests,
  formattedVectorTests,
  formattedMatrixTests
} = require('./tests/dev-tests')

const {
  testBp
} = require('./tests/bp-tests')

runAll()
testBp()