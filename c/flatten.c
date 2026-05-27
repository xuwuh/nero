#include <stdio.h>

#include "../include/flatten.h"

//инициализация
int flatten_init(Flatten *lauer, Tensor *input){
    lauer->input_n=input->n;
    lauer->input_c=input->c;
    lauer->input_h=input->h;
    lauer->input_w=input->w;
    lauer->output_features=input->c*input->h*input->w;
    return 1;
}

//прямой проход
int flatten_forward(Flatten *lauer, Tensor *input, Matrix *output){
    for (int n=0; n<input->n; n++){
        int feature_i=0;
        for (int c=0; c<input->c; c++){
            for (int h=0; h<input->h; h++){
                for (int w=0; w<input->w; w++){
                    double value=tensor_get(input, n,c, h, w);
                    matrix_set(output, n, feature_i, value);
                    feature_i++;
                }
            }
        }
    }
    return 1;
}

//обратный проход
int flatten_backward(Flatten *lauer, Matrix *doutput, Tensor *dinput){
    for (int n=0; n<dinput->n; n++){
        int feature_i=0;
        for (int c=0; c<dinput->c; c++){
            for (int h=0; h<dinput->h; h++){
                for (int w=0; w<dinput->w; w++){
                    double grad=matrix_get(doutput, n, feature_i);
                    tensor_set(dinput, n, c, h, w, grad);
                    feature_i++;
                }
            }
        }
    }
    return 1;
}

int flatten_test(){
    Tensor input;
    Tensor dinput;
    Matrix output;
    Matrix doutput;
    Flatten flatten;

    printf("\nFLATTEN TESTAA\n");

    input=tensor_create(1, 2, 2, 2);
    tensor_set(&input, 0, 0, 0, 0, 1.0);
    tensor_set(&input, 0, 0, 0, 1, 2.0);
    tensor_set(&input, 0, 0, 1, 0, 3.0);
    tensor_set(&input, 0, 0, 1, 1, 4.0);
    tensor_set(&input, 0, 1, 0, 0, 5.0);
    tensor_set(&input, 0, 1, 0, 1, 6.0);
    tensor_set(&input, 0, 1, 1, 0, 7.0);
    tensor_set(&input, 0, 1, 1, 1, 8.0);
    flatten_init(&flatten, &input);
    output=matrix_create(1, flatten.output_features);
    flatten_forward(&flatten, &input, &output);
    tensor_print(&input, "flatten input");
    matrix_print(&output, "flatten output");

    doutput=matrix_create(1, flatten.output_features);
    matrix_set(&doutput, 0, 0, 1.0);
    matrix_set(&doutput, 0, 1, 2.0);
    matrix_set(&doutput, 0, 2, 3.0);
    matrix_set(&doutput, 0, 3, 4.0);
    matrix_set(&doutput, 0, 4, 5.0);
    matrix_set(&doutput, 0, 5, 6.0);
    matrix_set(&doutput, 0, 6, 7.0);
    matrix_set(&doutput, 0, 7, 8.0);
    dinput=tensor_create(1, 2, 2, 2);

    flatten_backward(&flatten, &doutput, &dinput);
    tensor_print(&dinput, "flatten dinput");
    printf("flatten dinput values:\n");
    printf("channel 0:\n");
    printf("%.2f %.2f\n",tensor_get(&dinput, 0, 0, 0, 0),tensor_get(&dinput, 0, 0, 0, 1));
    printf("%.2f %.2f\n",tensor_get(&dinput, 0, 0, 1, 0),tensor_get(&dinput, 0, 0, 1, 1));
    printf("channel 1:\n");
    printf("%.2f %.2f\n",tensor_get(&dinput, 0, 1, 0, 0),tensor_get(&dinput, 0, 1, 0, 1));
    printf("%.2f %.2f\n",tensor_get(&dinput, 0, 1, 1, 0),tensor_get(&dinput, 0, 1, 1, 1));

    tensor_free(&input);
    tensor_free(&dinput);
    matrix_free(&output);
    matrix_free(&doutput);
    return 1;
}