#include <stdio.h>

#include "../include/relu.h"

//проверяем имеют ли тензоры одинаковый размер
int tensor_same_shape(Tensor *a, Tensor *b){
    if (a==NULL || b==NULL){return 0;}
    if (a->data==NULL || b->data==NULL){return 0;}
    if (a->n!=b->n){ return 0;}
    if (a->c!=b->c){return 0;}
    if (a->h!=b->h){return 0;}
    if (a->w!=b->w){return 0;}
    return 1;
}

//прямой проход: output[i] = max(0, input[i])
int relu_forward(Tensor *input, Tensor *output){
    if (!tensor_same_shape(input, output)){return 0;}
    for (int i=0; i<input->size; i++){
        if (input->data[i]>0.0){
            output->data[i]=input->data[i];
        }
        else {output->data[i]=0.0;}
    }
    return 1;
}

//оратный проход: dinput = doutput, если input > 0; dinput = 0, если input <= 0
int relu_backward(Tensor *input, Tensor *doutput, Tensor *dinput){
    if (!tensor_same_shape(input, doutput) || !tensor_same_shape(input, dinput)){return 0;}
    for (int i=0; i<input->size; i++){
        if (input->data[i]>0.0){
            dinput->data[i]=doutput->data[i];
        }
        else {dinput->data[i]=0.0;}
    }
    return 1;
}
