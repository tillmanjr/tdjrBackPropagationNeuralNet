'use strict;'

const isLessThan = (x, lessThan) => Boolean(x < lessThan)
const isGreaterThan = (x, greaterThan) => Boolean(x > greaterThan)

// Pair of inclusive bounds
function Bounds(lower, upper) {
    this.lowerBound = lower
    this.upperBound = upper,
    this.isBelowBounds = function (x) { return isLessThan(x, this.lowerBound ) },
    this.isAboveBounds = function (x) { return isGreaterThan( x, this.upperBound ) }
}

const createBounds = (lower, upper) => new Bounds(lower, upper)

module.exports = {
  createBounds: createBounds
}