#ifndef CONV_H
#define CONV_H

#include "tensor.h"
#include "matrix.h"

typedef struct
{
    int in_c; //каналы входа
    int out_c; //чильтры в кофиге
    int kernel_h; //высота фильтра
    int kernel_w; //ширина фильстра
    int stride; //шаг смещения филтра
    int padding; //рамка вокруг изображения (нулевая)

    Matrix weights; //значения всех фильтров
    Matrix bias; //смещение всех фисьлтро
    Matrix dweights; //градиенты по филтрам
    Matrix dbias; //градиенты со смещением
    Matrix velocity_weights; //помять моментум для фильтров
    Matrix velocity_bias; //помаять моментум для смещения 
} Conv;

//инициализация слоя
int conv_init(Conv *layer, int in_channels, int out_channels, int kernel_h, int kernel_w, int stride, int padding);
//прямой проход
int conv_forward(Conv *layer, Tensor *input, Tensor *output);
//обратный проход
int conv_backward(Conv *layer, Tensor *input, Tensor *doutput, Tensor *dinput);
//градиент
int conv_sgd(Conv *layer, double learning_rate, double regularization);
//градиент с моментум
int conv_momentum(Conv *layer, double learning_rate, double momentum, double regularization);

void conv_free(Conv *layer);
void conv_tests();
void conv_backward_tests();

#endif