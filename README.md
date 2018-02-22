# tdjrBackPropagationNeuralNet
Simple straightforward JavaScript neural network to explore implementing Back Propagation

`node ./index.js` tests the various libs, then builds a simple neural network and trains it with a very simple scenario

Caveat:
This is part ongoing work to teach myself the internals of certain ML appropaches.
In this case I create a simple 3-4-2 Neural Net which I train using Back Propagation. 

The purpose is to play with Back Propagation not neural nets. 
As such I have taken a few shortcuts:  
*Vectors*  
in this context Vector refers to a one dimensional value as opposed to a Scalar, a 0 dimensional value.
Vector = a one dimensional array of floating point numbers
  
*Matrix*  
I have implemented matrices as array[row index][column value] instead of a true matrix.
In practice this works well since it allows intuitive referencing.  
  
*Tests*  
There are no formal unit tests in this playground :( The tests emit text which the handler functions sent to stdout.  

Again the purpose is play so the test are really a rig for displaying the results of "play"
