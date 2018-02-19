"use strict;"

class BaseBackPropagationNeuralNetError {
  constructor (message) {
    this._message = message ? message : this.getName()
  }

  getMessage () { return this._message }

  /* this should really be a static method returning the instance's type as string */
  getName () { return 'BaseBackPropagationNeuralNetError'}

}


class InvalidModelException extends BaseBackPropagationNeuralNetError {
  constructor (message) {
    super (message)
  }

  getName () { return 'InvalidModelException' }
}

const exceptionsExports = {
  BaseBackPropagationNeuralNetError,
  InvalidModelException
}