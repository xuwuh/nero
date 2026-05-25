#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "matrix.h"
#include "optimizer.h"

typedef struct
{
    int input_size; //сколько признаков получает обьект
    int output_size; //сколько значений выдает слой
    Matrix weights; //веса, обучаемые связи между слоями
    Matrix bias; //смещение нейронов
    Matrix dweights; //градиент ошибки по весам
    Matrix dbias; //градиент ошибки по смещению
    Matrix velocity_weights; //память моментум для весов 
    Matrix velocity_bias; //помять моментум для смещения
} LinearLayer;

//инициализация слоя
int linear_init(LinearLayer *layer, int input_size, int output_size);
//прямой проход
int linear_forward(LinearLayer *layer, Matrix *input, Matrix *output);
//обратный проход
int linear_backward(LinearLayer *layer, Matrix *input, Matrix *doutput, Matrix *dinput);

//обновление весов градиентом
int linear_sgd(LinearLayer *layer, double learning_rate, double regularization);
//обновление весов градиент+моментум
int linear_sgd_momentum(LinearLayer *layer, double learning_rate,double momentum, double regularization);

//очищение памяти и тест 
void linear_free(LinearLayer *layer);
void linear_test();

#endif