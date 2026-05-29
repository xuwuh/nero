#include <stdio.h>
#include <stdlib.h>

#include "../include/model.h"
#include "../include/im2col.h"
#include "../include/relu.h"
#include "../include/loss.h"

//собираем слои в одну модель: Conv -> ReLU -> Pool -> flaten -> Linear
int model_init(Model *model, const Config *config){
    if (model==NULL || config==NULL){return 0;}

    model->input_c=config->input_ch;
    model->input_h=config->input_h;
    model->input_w=config->input_w;
    model->num_classes=config->classes;

    //инициализируем Conv
    if (!conv_init(&model->conv, config->input_ch, config->conv_out_channels, config->conv_kernel_h, config->conv_kernel_w, config->conv_stride, config->conv_padding)){return 0;}
    //считаем размер после
    model->conv_out_h=conv_output_size(config->input_h, config->conv_kernel_h, config->conv_stride, config->conv_padding);
    model->conv_out_w=conv_output_size(config->input_w, config->conv_kernel_w, config->conv_stride, config->conv_padding);
    if (model->conv_out_h<=0 || model->conv_out_w<=0){conv_free(&model->conv); return 0;}

    //инициализируем Pool
    if (!maxpool_init(&model->poll,config->pool_size, config->pool_stride)){conv_free(&model->conv); return 0;}
    //сотмрим размеры после 
    model->pool_out_h=maxpool_output_size(model->conv_out_h, config->pool_size, config->pool_stride);
    model->pool_out_w=maxpool_output_size(model->conv_out_w, config->pool_size, config->pool_stride);
    if (model->pool_out_h<=0 || model->pool_out_w<=0){conv_free(&model->conv); return 0;}

    //проходимся по flaten
    model->flatten_size=config->conv_out_channels*model->pool_out_h*model->pool_out_w;
    //фейковый тензор для тестов
    Tensor fake_pool_output=tensor_create(1, config->conv_out_channels, model->pool_out_h, model->pool_out_w);
    flatten_init(&model->flaten, &fake_pool_output);
    tensor_free(&fake_pool_output);

    //инициализируем Linear (финальный слой)
    if (!linear_init(&model->final_layer, model->flatten_size, config->classes)){conv_free(&model->conv); return 0;}

    printf("Model initialized:\n");
    printf("input: %d x %d x %d\n", model->input_c, model->input_h, model->input_w);
    printf("conv output: %d x %d x %d\n", config->conv_out_channels, model->conv_out_h, model->conv_out_w);
    printf("pool output: %d x %d x %d\n", config->conv_out_channels, model->pool_out_h, model->pool_out_w);
    printf("flaten size: %d\n", model->flatten_size);
    printf("classes: %d\n\n", model->num_classes);
    return 1;
}

//прямая проходка
int model_forward(Model *model, Tensor *input, Tensor *conv_out,Tensor *relu_out, Tensor *pool_out, Matrix *flat_out, Matrix *logits, Matrix *probabilities){
    if (!conv_forward(&model->conv, input, conv_out)){return 0;}
    if (!relu_forward(conv_out, relu_out)){return 0;}
    if (!maxpool_forward(&model->poll, relu_out, pool_out)){return 0;}
    if (!flatten_forward(&model->flaten, pool_out, flat_out)){return 0;}
    if (!linear_forward(&model->final_layer, flat_out, logits)){return 0;}
    if (!softmax(logits, probabilities)){return 0;}
    return 1;
}

//обратная проходка
int model_backward(Model *model, Tensor *input, Tensor *conv_out, Tensor *relu_out, Tensor *pool_out, Matrix *flat_out, Matrix *dlogits){
    Matrix dflat=matrix_create(flat_out->rows, flat_out->cols);
    Tensor dpool=tensor_create(pool_out->n, pool_out->c, pool_out->h, pool_out->w);
    Tensor drelu=tensor_create(relu_out->n, relu_out->c, relu_out->h, relu_out->w);
    Tensor dconv=tensor_create(conv_out->n, conv_out->c, conv_out->h, conv_out->w);
    Tensor dinput=tensor_create(input->n, input->c, input->h, input->w);

    if (!linear_backward(&model->final_layer, flat_out, dlogits, &dflat)){
        matrix_free(&dflat);
        tensor_free(&dpool);
        tensor_free(&drelu);
        tensor_free(&dconv);
        tensor_free(&dinput);
        return 0;
    }
    if (!flatten_backward(&model->flaten, &dflat, &dpool)){
        matrix_free(&dflat);
        tensor_free(&dpool);
        tensor_free(&drelu);
        tensor_free(&dconv);
        tensor_free(&dinput);
        return 0;
    }
    if (!maxpool_backward(&model->poll, relu_out, &dpool, &drelu)){
        matrix_free(&dflat);
        tensor_free(&dpool);
        tensor_free(&drelu);
        tensor_free(&dconv);
        tensor_free(&dinput);
        return 0;
    }
    if (!relu_backward(conv_out, &drelu, &dconv)){
        matrix_free(&dflat);
        tensor_free(&dpool);
        tensor_free(&drelu);
        tensor_free(&dconv);
        tensor_free(&dinput);
        return 0;
    }
    if (!conv_backward(&model->conv, input, &dconv, &dinput)){
        matrix_free(&dflat);
        tensor_free(&dpool);
        tensor_free(&drelu);
        tensor_free(&dconv);
        tensor_free(&dinput);
        return 0;
    }

    matrix_free(&dflat);
    tensor_free(&dpool);
    tensor_free(&drelu);
    tensor_free(&dconv);
    tensor_free(&dinput);
    return 1;
}

//обновление весов обычным градиентом
int model_sgd(Model *model, double learning_rate, double regularization){
    if (model==NULL){return 0;}
    if (!conv_sgd(&model->conv, learning_rate, regularization)){return 0;}
    if (!linear_sgd(&model->final_layer, learning_rate, regularization)){return 0;}
    return 1;
}

//обновление весов градиентом с моментум
int model_momentum(Model *model, double learning_rate, double momentum, double regularization){
    if(model==NULL){return 0;}
    if(!conv_momentum(&model->conv, learning_rate, momentum, regularization)){return 0;}
    if(!linear_sgd_momentum(&model->final_layer, learning_rate, momentum, regularization)){return 0;}
    return 1;
}

void model_free(Model *model){
    conv_free(&model->conv);
    linear_free(&model->final_layer);
    model->input_c=0;
    model->input_h=0;
    model->input_w=0;
    model->conv_out_h=0;
    model->conv_out_w=0;
    model->pool_out_h=0;
    model->pool_out_w=0;
    model->flatten_size=0;
    model->num_classes=0;
}
