#include <stdio.h>

#include "../include/optimizer.h"

//создаем оптимизатор
Optimizer optimizer_create(double learning_rate, double momentum, double regularizator){
    Optimizer optimizer;
    optimizer.learning_rate=learning_rate;
    optimizer.momentum=momentum;
    optimizer.regularizator=regularizator;
    return optimizer;
}

//градиентный спуск: weights = weights - learning_rate * gradients
int optimizer_sgd(Matrix *weights, Matrix *gradient, double learning_rate){
    if (weights==NULL || gradient==NULL || weights->rows!=gradient->rows || weights->cols!=gradient->cols){return 0;}
    int total=weights->rows*weights->cols;
    for (int i=0; i<total; i++){
        weights->data[i]=weights->data[i]-learning_rate*gradient->data[i];
    }
    return 1;
}

//спуск со штрафом: weights = weights - learning_rate * (gradients + regularization * weights)
int optimizer_sgd_l2(Matrix *weights, Matrix *gradient, double learning_rate, double regularizator){
    if (weights==NULL || gradient==NULL || weights->rows!=gradient->rows || weights->cols!=gradient->cols){return 0;}
    int total=weights->rows*weights->cols;
    for (int i=0; i<total; i++){
        double l2_gradien=regularizator*weights->data[i];
        weights->data[i]=weights->data[i]-learning_rate*(gradient->data[i]+l2_gradien);
    }
    return 1;
}

//спуск с моментум: weights = weights + velocity || velocity = momentum * velocity - learning_rate * gradients
int optimizer_sgd_momentum(Matrix *weights, Matrix *gradient,Matrix *velocity, double learning_rate, double momentum){
     if (weights==NULL || gradient==NULL || weights->rows!=gradient->rows || weights->cols!=gradient->cols || 
        velocity == NULL || weights->rows!=velocity->rows || weights->cols!=velocity->cols){return 0;}    
    int total=weights->rows*weights->cols;
    for (int i=0; i<total; i++){
        velocity->data[i]=momentum*velocity->data[i]-learning_rate*gradient->data[i];
        weights->data[i]=weights->data[i]+velocity->data[i];
    }
    return 1;
}

//спуск со штрафом и моментум: gradient_total = gradients + regularization * weights || velocity = momentum * velocity - learning_rate * gradient_total || weights = weights + velocity
int optimizer_sgd_momentum_l2(Matrix *weights, Matrix *gradient,Matrix *velocity, double learning_rate, double momentum, double regularizator){
     if (weights==NULL || gradient==NULL || weights->rows!=gradient->rows || weights->cols!=gradient->cols || 
        velocity == NULL || weights->rows!=velocity->rows || weights->cols!=velocity->cols){return 0;} 
    int total=weights->rows*weights->cols;
    for (int i=0; i<total; i++){
        double gradient_total=gradient->data[i]+regularizator*weights->data[i];
        velocity->data[i]=momentum*velocity->data[i]-learning_rate*gradient_total;
        weights->data[i]=weights->data[i]+velocity->data[i];
    }
    return 1;
}
