"use strict;"

const NEW_LINE_CHAR = '\n'


function formatVector (
  vectorArray,
  fixedDecimals,
  valuesPerLine,
  appendBlankLine
) {
  let result = '';

  console.log('vectorArray : ', vectorArray)

  for (var i = 0; i < vectorArray.length; ++i) {
    if (i > 0 && i % valuesPerLine == 0) // max of 12 values per row
      result += NEW_LINE_CHAR

    const val = vectorArray[i]
    if (val >= 0.0) result += (' ');

    result += `${val.toFixed(fixedDecimals)} `
  }

  if (appendBlankLine)  result += NEW_LINE_CHAR

  return result
}

function formatMatrix (
  matrix, // [row index 0..n][row values]
  numRows,
  fixedDecimals
) {
  let result = ''

  let ct = 0
  if (numRows == -1) numRows = Number.MAX_SAFE_INTEGER; // if numRows == -1, show all rows
  for (var i = 0; i < matrix.length && ct < numRows; ++i)
  {
    for (var j = 0; j < matrix[0].length; ++j)
    {
      const val = matrix[i][j]
      if (val >= 0.0) result += ' ' // blank space instead of '+' sign
      result += `${val.toFixed(fixedDecimals)} `
    }

    ct ++
    result += NEW_LINE_CHAR
  }

  result += NEW_LINE_CHAR

  return result;
}

module.exports = {
  formatVector,
  formatMatrix
}
