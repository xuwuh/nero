#include <stdio.h>
#include <math.h>

#include "../include/conv.h"
#include "../include/im2col.h"
#include "../include/utils.h"
#include "../include/optimizer.h"

//размер слоя
void conv_init_weights(Matrix *weights){
    int total=weights->rows*weights->cols;
    double scale=sqrt(2.0/(weights->rows+weights->cols));
    for (int i=0; i<total; i++){
        weights->data[i]=random_weight(scale);
    }
}

//инициализация слоя
int conv_init(Conv *layer, int in_c, int out_c, int kernel_h, int kernel_w, int stride, int padding){
    layer->in_c=in_c;
    layer->out_c=out_c;
    layer->kernel_h=kernel_h;
    layer->kernel_w=kernel_w;
    layer->stride=stride;
    layer->padding=padding;

    int weight_rows=in_c*kernel_h*kernel_w;
    int weight_cols=out_c;
    
    layer->weights=matrix_create(weight_rows, weight_cols);
    layer->bias=matrix_create(1, out_c);
    layer->dweights=matrix_create(weight_rows, weight_cols);
    layer->dbias=matrix_create(1, out_c);
    layer->velocity_weights=matrix_create(weight_rows, weight_cols);
    layer->velocity_bias=matrix_create(1, out_c);

    conv_init_weights(&layer->weights);
    matrix_zero(&layer->bias);
    matrix_zero(&layer->dweights);
    matrix_zero(&layer->dbias);
    matrix_zero(&layer->velocity_weights);
    matrix_zero(&layer->velocity_bias);
    return 1;
}

void conv_output_col_to_tencor(Matrix *output_col, Tensor *output){
    for (int n=0; n<output->n; n++){
        for (int h=0; h<output->h; h++){
            for (int w=0; w<output->w; w++){
                int row_i=n*output->h*output->w+h*output->w+w;
                for (int c=0; c<output->c; c++){
                    tensor_set(output, n,c,h,w, matrix_get(output_col, row_i, c));
                }
            }
        }
    }
}

void conv_doutput_tensor_to_col(Tensor *doutput, Matrix *doutput_col){
    for (int n=0; n<doutput->n; n++){
        for (int h=0; h<doutput->h; h++){
            for (int w=0; w<doutput->w; w++){
                int row_i=n*doutput->h*doutput->w+h*doutput->w+w;
                for (int c=0; c<doutput->c; c++){
                    matrix_set(doutput_col, row_i, c, tensor_get(doutput, n, c, h, w));
                }
            }
        }
    }
}

//прямой проход
int conv_forward(Conv *layer, Tensor *input, Tensor *output){
    if (layer==NULL || input==NULL || output==NULL){return 0;}
    int out_h=conv_output_size(input->h, layer->kernel_h, layer->stride, layer->padding);
    int out_w=conv_output_size(input->w, layer->kernel_w, layer->stride, layer->padding);
    if (input->c!=layer->in_c || output->n!=input->n || output->c!=layer->out_c || output->h!=out_h || output->w!=out_w){return 0;}

    int col_rows=input->n*output->h*output->w;
    int col_cols=layer->in_c*layer->kernel_h*layer->kernel_w;
    Matrix input_col=matrix_create(col_rows, col_cols);
    Matrix output_col=matrix_create(col_rows, layer->out_c);

    if (!im2col(input, &input_col, layer->kernel_h, layer->kernel_w, layer->stride, layer->padding)){
        matrix_free(&input_col); matrix_free(&output_col); return 0;
    }
    if (!matrix_mult(&input_col, &layer->weights, &output_col)){
        matrix_free(&input_col); matrix_free(&output_col); return 0;
    }
    if (!matrix_dias(&output_col, &layer->bias)){
        matrix_free(&input_col); matrix_free(&output_col); return 0;
    }
    conv_output_col_to_tencor(&output_col, output);
    matrix_free(&input_col);
    matrix_free(&output_col);
    return 1;
}

//обратный проход
int conv_backward(Conv *layer, Tensor *input, Tensor *doutput, Tensor *dinput){
    if (layer==NULL || dinput==NULL || doutput==NULL){return 0;}
    int out_h=conv_output_size(dinput->h, layer->kernel_h, layer->stride, layer->padding);
    int out_w=conv_output_size(dinput->w, layer->kernel_w, layer->stride, layer->padding);
    if (dinput->c!=layer->in_c || doutput->n!=dinput->n || doutput->c!=layer->out_c || doutput->h!=out_h || doutput->w!=out_w){return 0;}

    int col_rows=input->n*doutput->h*doutput->w;
    int col_cols=layer->in_c*layer->kernel_h*layer->kernel_w;
    Matrix input_col=matrix_create(col_rows, col_cols);
    Matrix input_col_t=matrix_create(col_cols, col_rows);
    Matrix doutput_col=matrix_create(col_rows, layer->out_c);
    Matrix weights_t=matrix_create(layer->out_c, col_cols);
    Matrix dinput_col=matrix_create(col_rows, col_cols);

    im2col(input, &input_col, layer->kernel_h, layer->kernel_w, layer->stride, layer->padding);
    conv_doutput_tensor_to_col(doutput, &doutput_col);

    matrix_transpose(&input_col, &input_col_t);
    matrix_mult(&input_col_t, &doutput_col, &layer->dweights);
    matrix_zero(&layer->dbias);

    for (int j=0; j<doutput_col.cols; j++){
        double sum=0.0;
        for (int i=0; i<doutput_col.rows; i++){sum+=matrix_get(&doutput_col, i, j);}
        matrix_set(&layer->dbias, 0, j, sum);
    }

    matrix_transpose(&layer->weights, &weights_t);
    matrix_mult(&doutput_col, &weights_t, &dinput_col);

    col2im(&dinput_col, dinput, layer->kernel_h, layer->kernel_w, layer->stride, layer->padding);

    matrix_free(&input_col);
    matrix_free(&input_col_t);
    matrix_free(&doutput_col);
    matrix_free(&weights_t);
    matrix_free(&dinput_col);
    return 1;
}

//градиент
int  conv_sgd(Conv *layer, double learning_rate, double regularization){
    if(!optimizer_sgd_l2(&layer->weights, &layer->dweights, learning_rate, regularization)){return 0;}
    if(!optimizer_sgd_l2(&layer->bias, &layer->dbias, learning_rate, 0.0)){return 0;}
    return 1;
}

//градиент с моментум
int conv_momentum(Conv *layer, double learning_rate, double momentum, double regularization){
    if(!optimizer_sgd_momentum_l2(&layer->weights, &layer->dweights, &layer->velocity_weights, learning_rate, momentum, regularization)){return 0;}
    if(!optimizer_sgd_momentum_l2(&layer->bias, &layer->dbias, &layer->velocity_bias, learning_rate, momentum, 0.0)){return 0;}
    return 1;
}

void conv_free(Conv *layer){
    matrix_free(&layer->weights);
    matrix_free(&layer->bias);
    matrix_free(&layer->dweights);
    matrix_free(&layer->dbias);
    matrix_free(&layer->velocity_weights);
    matrix_free(&layer->velocity_bias);

    layer->in_c=0;
    layer->out_c=0;
    layer->kernel_h=0;
    layer->kernel_w=0;
    layer->stride=0;
    layer->padding=0;
}


void conv_tests(){
    Conv conv;
    Tensor input;
    Tensor output;

    printf("\nCONV FORVARD TEST\n");
    conv_init(&conv, 1, 1, 2, 2, 1, 0);

    matrix_set(&conv.weights, 0, 0, 1.0);
    matrix_set(&conv.weights, 1, 0, 0.0);
    matrix_set(&conv.weights, 2, 0, 0.0);
    matrix_set(&conv.weights, 3, 0, 1.0);
    matrix_set(&conv.bias, 0, 0, 0.0);

    input=tensor_create(1, 1, 3, 3);
    output=tensor_create(1, 1, 2, 2);

    tensor_set(&input, 0, 0, 0, 0, 1.0);
    tensor_set(&input, 0, 0, 0, 1, 2.0);
    tensor_set(&input, 0, 0, 0, 2, 3.0);
    tensor_set(&input, 0, 0, 1, 0, 4.0);
    tensor_set(&input, 0, 0, 1, 1, 5.0);
    tensor_set(&input, 0, 0, 1, 2, 6.0);
    tensor_set(&input, 0, 0, 2, 0, 7.0);
    tensor_set(&input, 0, 0, 2, 1, 8.0);
    tensor_set(&input, 0, 0, 2, 2, 9.0);

    conv_forward(&conv, &input, &output);

    tensor_print(&input, "conv input");
    tensor_print(&output, "conv output");
    printf("Conv output values:\n");
    printf("%.2f %.2f\n",tensor_get(&output, 0, 0, 0, 0),tensor_get(&output, 0, 0, 0, 1));
    printf("%.2f %.2f\n",tensor_get(&output, 0, 0, 1, 0),tensor_get(&output, 0, 0, 1, 1));

    tensor_free(&input);
    tensor_free(&output);
    conv_free(&conv);
}

void conv_backward_tests(){
    Conv conv;
    Tensor input;
    Tensor output;
    Tensor doutput;
    Tensor dinput;

    printf("\nCONV BACKWARD TEST\n");

    conv_init(&conv, 1, 1, 2, 2, 1, 0);

    matrix_set(&conv.weights, 0, 0, 1.0);
    matrix_set(&conv.weights, 1, 0, 0.0);
    matrix_set(&conv.weights, 2, 0, 0.0);
    matrix_set(&conv.weights, 3, 0, 1.0);
    matrix_set(&conv.bias, 0, 0, 0.0);

    input=tensor_create(1, 1, 3, 3);
    output=tensor_create(1, 1, 2, 2);
    doutput=tensor_create(1, 1, 2, 2);
    dinput=tensor_create(1, 1, 3, 3);

    tensor_set(&input, 0, 0, 0, 0, 1.0);
    tensor_set(&input, 0, 0, 0, 1, 2.0);
    tensor_set(&input, 0, 0, 0, 2, 3.0);
    tensor_set(&input, 0, 0, 1, 0, 4.0);
    tensor_set(&input, 0, 0, 1, 1, 5.0);
    tensor_set(&input, 0, 0, 1, 2, 6.0);
    tensor_set(&input, 0, 0, 2, 0, 7.0);
    tensor_set(&input, 0, 0, 2, 1, 8.0);
    tensor_set(&input, 0, 0, 2, 2, 9.0);

    conv_forward(&conv, &input, &output);
    tensor_value(&doutput, 1.0);
    conv_backward(&conv, &input, &doutput, &dinput);

    matrix_print(&conv.dweights, "conv dweights");
    matrix_print(&conv.dbias, "conv dbias");
    tensor_print(&dinput, "conv dinput");

    printf("conv dinput values:\n");
    printf("%.2f %.2f %.2f\n",tensor_get(&dinput, 0, 0, 0, 0),tensor_get(&dinput, 0, 0, 0, 1),tensor_get(&dinput, 0, 0, 0, 2));
    printf("%.2f %.2f %.2f\n",tensor_get(&dinput, 0, 0, 1, 0),tensor_get(&dinput, 0, 0, 1, 1),tensor_get(&dinput, 0, 0, 1, 2));
    printf("%.2f %.2f %.2f\n",tensor_get(&dinput, 0, 0, 2, 0),tensor_get(&dinput, 0, 0, 2, 1),tensor_get(&dinput, 0, 0, 2, 2));

    tensor_free(&input);
    tensor_free(&output);
    tensor_free(&doutput);
    tensor_free(&dinput);
    conv_free(&conv);
}
