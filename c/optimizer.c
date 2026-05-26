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


//test
void optimizer_test(){
    Matrix weights_sgd;
    Matrix weights_momentum;
    Matrix weights_l2;
    Matrix weights_momentum_l2;
    Matrix gradients;
    Matrix velocity;
    Matrix velocity_l2;

    double learning_rate = 0.1;
    double momentum = 0.9;
    double regularizator = 0.01;

    printf("\nOPTIMIZER TEST\n");

    weights_sgd = matrix_create(2, 2);
    matrix_set(&weights_sgd, 0, 0, 1.0);
    matrix_set(&weights_sgd, 0, 1, 2.0);
    matrix_set(&weights_sgd, 1, 0, 3.0);
    matrix_set(&weights_sgd, 1, 1, 4.0);

    weights_momentum = matrix_create(2, 2);
    weights_l2 = matrix_create(2, 2);
    weights_momentum_l2 = matrix_create(2, 2);

    matrix_copy(&weights_sgd, &weights_momentum);
    matrix_copy(&weights_sgd, &weights_l2);
    matrix_copy(&weights_sgd, &weights_momentum_l2);

    gradients = matrix_create(2, 2);
    matrix_set(&gradients, 0, 0, 0.1);
    matrix_set(&gradients, 0, 1, 0.2);
    matrix_set(&gradients, 1, 0, 0.3);
    matrix_set(&gradients, 1, 1, 0.4);

    velocity = matrix_create(2, 2);
    velocity_l2 = matrix_create(2, 2);
    matrix_zero(&velocity);
    matrix_zero(&velocity_l2);

    matrix_print(&weights_sgd, "initial weights");
    matrix_print(&gradients, "gradients");

    optimizer_sgd(&weights_sgd, &gradients, learning_rate);
    matrix_print(&weights_sgd, "weights after SGD");

    optimizer_sgd_momentum(
        &weights_momentum, &gradients, &velocity,
        learning_rate, momentum
    );
    matrix_print(&velocity, "velocity after Momentum step 1");
    matrix_print(&weights_momentum, "weights after Momentum step 1");

    optimizer_sgd_momentum(
        &weights_momentum, &gradients, &velocity,
        learning_rate, momentum
    );
    matrix_print(&velocity, "velocity after Momentum step 2");
    matrix_print(&weights_momentum, "weights after Momentum step 2");

    optimizer_sgd_l2(
        &weights_l2, &gradients,
        learning_rate, regularizator
    );
    matrix_print(&weights_l2, "weights after SGD + L2");

    optimizer_sgd_momentum_l2(
        &weights_momentum_l2, &gradients, &velocity_l2,
        learning_rate, momentum, regularizator
    );
    matrix_print(&velocity_l2, "velocity after Momentum + L2 step 1");
    matrix_print(&weights_momentum_l2, "weights after Momentum + L2 step 1");

    optimizer_sgd_momentum_l2(
        &weights_momentum_l2, &gradients, &velocity_l2,
        learning_rate, momentum, regularizator
    );
    matrix_print(&velocity_l2, "velocity after Momentum + L2 step 2");
    matrix_print(&weights_momentum_l2, "weights after Momentum + L2 step 2");

    matrix_free(&weights_sgd);
    matrix_free(&weights_momentum);
    matrix_free(&weights_l2);
    matrix_free(&weights_momentum_l2);
    matrix_free(&gradients);
    matrix_free(&velocity);
    matrix_free(&velocity_l2);
}
