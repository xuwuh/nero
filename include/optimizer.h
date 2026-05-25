#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"

typedef struct
{
    double learning_rate; //скорость обучения 
    double momentum; //коэфициээнт моментум
    double regularizator; //l2 регуляризация
} Optimizer;

//создаем оптимизатор
Optimizer optimizer_create(double learning_rate, double momentum, double regularizator);
//градиентный спуск 
int optimizer_sgd(Matrix *weights, Matrix *gradient, double learning_rate);
//спуск со штрафом
int optimizer_sgd_l2(Matrix *weights, Matrix *gradient, double learning_rate, double regularizator);
//спуск с моментум 
int optimizer_sgd_momentum(Matrix *weights, Matrix *gradient,Matrix *velocity, double learning_rate, double momentum);
//спуск со штрафом и моментум
int optimizer_sgd_momentum_l2(Matrix *weights, Matrix *gradient,Matrix *velocity, double learning_rate, double momentum, double regularizator);

//test
void optimizer_test();

#endif
