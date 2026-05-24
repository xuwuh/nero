#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

//делаем нормальные вероятности классов
int softmax(Matrix *logits, Matrix *probabilities);
//смотрим вероятность правильных классов и считаем ошибку
double cross_entripy(Matrix *probabilities, int *labels);
//смотрим как надо изменить выходы модели 
int backward (Matrix *probabilities, int *labels, Matrix *dlogits);

void loss_test();

#endif