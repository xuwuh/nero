#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "tensor.h"
#include "matrix.h"
#include "conv.h"
#include "maxpool.h"
#include "flatten.h"
#include "linear_layer.h"

typedef struct
{
    Conv conv;
    MaxPool poll;
    Flatten flaten;
    LinearLayer final_layer;

    int input_h;
    int input_w;
    int input_c;
    int num_classes;
    
    int conv_out_h;
    int conv_out_w;

    int pool_out_h;
    int pool_out_w;

    int flatten_size;
} Model;

//инициализация модели
int model_init(Model *model, const Config *config);
//прямая проходка
int model_forward(Model *model, Tensor *input, Tensor *conv_out, Tensor *relu_out, Tensor *pool_out, Matrix *flat_out, Matrix *logits, Matrix *probabilities);
//обратная проходка
int model_backward(Model *model, Tensor *input, Tensor *conv_out, Tensor *relu_out, Tensor *pool_out, Matrix *flat_out, Matrix *dlogits);
//обновление весов обычным градиентом
int model_sgd(Model *model, double learning_rate, double regularization);
//обновление весов градиентом с моментум
int model_momentum(Model *model, double learning_rate, double momentum, double regularization);

void model_free(Model *model);

#endif
