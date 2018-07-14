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

// use to ignore message
const logNullFn = (message) => {return 0}
// for displaying progress messages
const logMessageFn = (message) => console.log(message)
// for displaying the results of a test
const logResultsFn = (message) => console.log(message)

runAll()

// enable next line to see details of training run
// testBp(logResultsFn, logMessageFn)

// enable next line to see only results of training run
testBp(logResultsFn, logNullFn)