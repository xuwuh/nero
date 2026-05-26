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

//тесты
void relu_test(){
    Tensor input;
    Tensor output;
    Tensor doutput;
    Tensor dinput;

    printf("\nRELU TEST\n");

    input=tensor_create(1, 1, 2, 3);
    output=tensor_create(1, 1, 2, 3);
    doutput=tensor_create(1, 1, 2, 3);
    dinput=tensor_create(1, 1, 2, 3);
    tensor_set(&input, 0, 0, 0, 0, -2.0);
    tensor_set(&input, 0, 0, 0, 1, -1.0);
    tensor_set(&input, 0, 0, 0, 2, 0.0);
    tensor_set(&input, 0, 0, 1, 0, 1.0);
    tensor_set(&input, 0, 0, 1, 1, 2.0);
    tensor_set(&input, 0, 0, 1, 2, 3.0);

    tensor_value(&doutput, 1.0);
    relu_forward(&input, &output);
    relu_backward(&input, &doutput, &dinput);

    tensor_print(&input, "ReLU input");
    printf("input values:\n");
    printf("%.2f %.2f %.2f %.2f %.2f %.2f\n", input.data[0], input.data[1], input.data[2], input.data[3], input.data[4], input.data[5]);
    printf("ReLU output values:\n");
    printf("%.2f %.2f %.2f %.2f %.2f %.2f\n", output.data[0], output.data[1], output.data[2], output.data[3], output.data[4], output.data[5]);
    printf("ReLU backward dinput values:\n");
    printf("%.2f %.2f %.2f %.2f %.2f %.2f\n", dinput.data[0], dinput.data[1], dinput.data[2], dinput.data[3], dinput.data[4], dinput.data[5]);

    tensor_free(&input);
    tensor_free(&output);
    tensor_free(&doutput);
    tensor_free(&dinput);
}