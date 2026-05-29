#ifndef RELU_H
#define RELU_H

#include "tensor.h"

//прямой проход
int relu_forward(Tensor *input, Tensor *output);
//оратный проход
int relu_backward(Tensor *input, Tensor *doutput, Tensor *dinput);

#endif
