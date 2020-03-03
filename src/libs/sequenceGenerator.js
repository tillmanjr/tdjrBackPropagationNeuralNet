'use strict;'



/* Examples of usage
// ==========================================

// square a number, no checks
const sqr = (x) => x * x;

// generate a sequence of even numbers, 0 - 18
let myEvens = generateSequence(0, 10, 2)

// map over myEven to generate the square of each value
let mySquaredEvens = [...myEvens].map(sqr)

// sum two numbers
const sumTwoNumbers = (x, y) => x + y

// generate a sequence of odd numbers, 1 - 19
let myOdds = generateSequence(1, 10, 2)

// reduce myOdds to derive the sum the range elements
const mySum = [...myEvens].reduce(sumTwoNumbers)



*/

// simple sequence generator supporting yield and of cource @@Iterator & @@iterable
// doesn't support reset()
function* generateSequence(start, count, stepBy) {
  let nextVal = start
  const stop = start + (count * stepBy)
  while (nextVal < stop) {
    let current = nextVal;
    nextVal = current + stepBy
    yield current;
  }
  return(nextVal)
}

module.exports = {
  generateSequence
}