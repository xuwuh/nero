#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "tensor.h"

typedef struct
{
    int pool_size ; //размер окна
    int stride; //шаг окна
} MaxPool;

//инициализация слоя
int maxpool_init(MaxPool *layer, int pool_size, int stride);
//прямой проход
int maxpool_forward(MaxPool *layer, Tensor *input, Tensor *output);
//обратный проход
int maxpool_backward(MaxPool *layer, Tensor *input, Tensor *doutput, Tensor *dinput);
//подсчет размеров на выходе из слоя
int maxpool_output_size(int input_size, int pool_size, int stride);

#endif
