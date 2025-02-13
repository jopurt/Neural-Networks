# Neural-Networks
This repo contains 3 implementations of different Neural Network models(x + 2, and/or, xor/xnor).

## Structure
Each folder contains 2 model implementations. 

For the first one, Python and standard libraries were used. 

For the second one, PyTorch(CUDA) was used.

### 1.Linear model (x+2)
Implementation of basic linear model with 1 neuron.

### 2.Nonlinear model (and | or)
Implementation of basic nonlinear model with 2 inputs and 1 neuron.

### 3.Complex nonlinear model (xor | xnor)
Implementation of complex nonlinear model with 2 inputs, 2 neuron on first layer and 1 neuron second layer.

We can represent xor/xnor as: 

xor = (a OR b) AND (a AND b)

xnor = (a AND b) OR (NOT a AND NOT b)
