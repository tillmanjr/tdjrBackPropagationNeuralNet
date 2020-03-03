'use strict;'

function _zip(func, args) {
  const iterators = args.map(arr => arr[Symbol.iterator]());
  let iterateInstances = iterators.map((i) => i.next());
  ret = []
  while(iterateInstances[func](it => !it.done)) {
    ret.push(iterateInstances.map(it => it.value));
    iterateInstances = iterators.map((i) => i.next());
  }
  return ret;
}

const zipShort = (...args) => _zip('every', args);

const zipLong = (...args) => _zip('some', args);

module.exports = {
  zipShort,
  zipLong
}