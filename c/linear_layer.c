#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/linear_layer.h"
#include "../include/utils.h"

//инициализация весом маленьикими случайными числами
void linear_init_weight(Matrix *weight){
    int total=weight->rows*weight->cols;
    double scale=sqrt(2.0/(weight->rows+weight->cols));
    for (int i=0; i<total; i++){
        weight->data[i]=random_weight(scale);
    }
}

//инициализация слоя
int linear_init(LinearLayer *layer, int input_size, int output_size){
    layer->input_size=input_size;
    layer->output_size=output_size;
    layer->weights=matrix_create(input_size, output_size);
    layer->bias=matrix_create(1, output_size);
    layer->dweights=matrix_create(input_size, output_size);
    layer->dbias=matrix_create(1, output_size);
    layer->velocity_weights=matrix_create(input_size, output_size);
    layer->velocity_bias=matrix_create(1, output_size);
    linear_init_weight(&layer->weights);

    matrix_zero(&layer->bias);
    matrix_zero(&layer->dweights);
    matrix_zero(&layer->dbias);
    matrix_zero(&layer->velocity_weights);
    matrix_zero(&layer->velocity_bias);
    return 1;
}

//прямой проход
int linear_forward(LinearLayer *layer, Matrix *input, Matrix *output){
    if(!matrix_mult(input, &layer->weights, output)){return 0;} //output = input * weights
    if(!matrix_dias(output, &layer->bias)){return 0;} //добавляем биас
    return 1;
}

//обратный проход
int linear_backward(LinearLayer *layer, Matrix *input, Matrix *doutput, Matrix *dinput){
    if (doutput->rows!=input->rows || doutput->cols!=layer->output_size ||
    dinput->rows!=input->rows || dinput->cols!=layer->input_size){return 0;}
    
    Matrix input_t=matrix_create(input->cols, input->rows);
    matrix_transpose(input, &input_t);
    matrix_mult(&input_t, doutput, &layer->dweights);

    matrix_zero(&layer->dbias);
    for (int j=0; j<doutput->cols; j++){
        double sum=0.0;
        for (int i=0; i<doutput->rows; i++){
            sum+=matrix_get(doutput, i, j);
        }
        matrix_set(&layer->dbias, 0, j, sum);
    }

    Matrix weghts_t=matrix_create(layer->weights.cols, layer->weights.rows);
    matrix_transpose(&layer->weights, &weghts_t);
    matrix_mult(doutput, &weghts_t, dinput);

    matrix_free(&input_t);
    matrix_free(&weghts_t);
    return 1;
}

//обновление весов градиентом
int linear_sgd(LinearLayer *layer, double learning_rate, double regularization){
    optimizer_sgd_l2(&layer->weights, &layer->dweights, learning_rate, regularization);
    optimizer_sgd_l2(&layer->bias, &layer->dbias, learning_rate, 0.0);
    return 1;
}

//обновление весов градиент+моментум
int linear_sgd_momentum(LinearLayer *layer, double learning_rate,double momentum, double regularization){
    optimizer_sgd_momentum_l2(&layer->weights, &layer->dweights, &layer->velocity_weights, learning_rate, momentum, regularization);
    optimizer_sgd_momentum_l2(&layer->bias, &layer->dbias, &layer->velocity_bias, learning_rate, momentum, 0.0);
    return 1;
}

//очищение памяти и тест 
void linear_free(LinearLayer *layer){
    matrix_free(&layer->weights);
    matrix_free(&layer->bias);
    matrix_free(&layer->dweights);
    matrix_free(&layer->dbias);
    matrix_free(&layer->velocity_weights);
    matrix_free(&layer->velocity_bias);
    layer->input_size=0;
    layer->output_size=0;
}

void linear_test(){
    LinearLayer layer;
    Matrix input;
    Matrix output;
    Matrix doutput;
    Matrix dinput;

    printf("\nLINEAR LAYER TEST\n");
    srand(1);

    linear_init(&layer, 3, 2);
    input=matrix_create(2, 3);

    matrix_set(&input, 0, 0, 1.0);
    matrix_set(&input, 0, 1, 2.0);
    matrix_set(&input, 0, 2, 3.0);
    matrix_set(&input, 1, 0, 4.0);
    matrix_set(&input, 1, 1, 5.0);
    matrix_set(&input, 1, 2, 6.0);

    output=matrix_create(2, 2);
    linear_forward(&layer, &input, &output);
    matrix_print(&input, "linear input");
    matrix_print(&layer.weights, "linear weights");
    matrix_print(&layer.bias, "linear bias");
    matrix_print(&output, "linear output");

    doutput=matrix_create(2, 2);
    matrix_set(&doutput, 0, 0, 0.1);
    matrix_set(&doutput, 0, 1, -0.1);
    matrix_set(&doutput, 1, 0, 0.2);
    matrix_set(&doutput, 1, 1, -0.2);

    dinput=matrix_create(2, 3);
    linear_backward(&layer, &input, &doutput, &dinput);
    matrix_print(&doutput, "doutput");
    matrix_print(&layer.dweights, "dweights");
    matrix_print(&layer.dbias, "dbias");
    matrix_print(&dinput, "dinput");
    linear_sgd(&layer, 0.01, 0.0001);
    matrix_print(&layer.weights, "weights after SGD update");
    matrix_print(&layer.bias, "bias after SGD update");

    matrix_free(&input);
    matrix_free(&output);
    matrix_free(&doutput);
    matrix_free(&dinput);
    linear_free(&layer);
}