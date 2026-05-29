#ifndef FLATTEN_H
#define FLATTEN_H

#include "tensor.h"
#include "matrix.h"

typedef struct 
{
    int input_n; //
    int input_c; //
    int input_h; //
    int input_w; //
    int output_features; //
} Flatten;

//инициализация
int flatten_init(Flatten *lauer, Tensor *input);
//прямой проход
int flatten_forward(Flatten *lauer, Tensor *input, Matrix *output);
//обратный проход
int flatten_backward(Flatten *lauer, Matrix *doutput, Tensor *dinput);

#endif
