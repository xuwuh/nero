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
    if (lauer==NULL || input==NULL || output==NULL){return 0;}
    if (output->rows!=input->n || output->cols!=lauer->output_features){return 0;}

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
    if (doutput->rows!=dinput->n || doutput->cols!=lauer->output_features || dinput->c!=lauer->input_c || dinput->h!=lauer->input_h || dinput->w!=lauer->input_w){return 0;}

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
