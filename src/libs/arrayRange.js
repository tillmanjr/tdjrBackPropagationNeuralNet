'use strict;'

function createRange(start, end, step = 1) {
  const len = Math.floor((end - start) / step) + 1
  return Array(len).fill().map((_, idx) => start + (idx * step))
}

module.exports = {
  createRange: createRange
}