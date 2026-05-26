#include <stdio.h>
#include <math.h>
#define Epsilon 1e-16

#include "../include/loss.h"

//находим максимальное значение в строке матрицы
double row_max(Matrix *matrix, int row){
    double max_val=matrix_get(matrix, row, 0);
    for (int i=0; i<matrix->cols; i++){
        double value=matrix_get(matrix, row, i);
        if (value>max_val){max_val=value;}
    }
    return max_val;
}

//делаем нормальные вероятности классов
int softmax(Matrix *logits, Matrix *probabilities){
    if (logits==NULL || probabilities==NULL || logits->rows!=probabilities->rows || logits->cols!=probabilities->cols){return 0;}

    for (int i=0; i<logits->rows; i++){
        double max_val=row_max(logits, i); 
        double sum_exp=0.0;
        for (int j=0; j<logits->cols; j++){
            double shifted_logit=matrix_get(logits, i, j)-max_val;
            double exp_val=exp(shifted_logit);
            matrix_set(probabilities, i, j, exp_val);
            sum_exp+=exp_val;
        }
        for (int j=0; j<probabilities->cols; j++){
            double value=matrix_get(probabilities, i, j) / sum_exp;
            matrix_set(probabilities, i, j, value);
        }
    }
    return 1;
}

//смотрим вероятность правильных классов и считаем ошибку
double cross_entripy(Matrix *probabilities, int *labels){
    double loss_sum=0.0;
    for (int i=0; i<probabilities->rows; i++){
        if (labels[i]<0 || labels[i]>=probabilities->cols){return -1.0;}
        int true_class=labels[i];
        double probability=matrix_get(probabilities,i , true_class);
        loss_sum+= -log(probability+Epsilon);
    }
    return loss_sum/probabilities->rows;
}

//смотрим как надо изменить выходы модели 
int backward (Matrix *probabilities, int *labels, Matrix *dlogits){
    if (probabilities==NULL || labels==NULL || dlogits==NULL || probabilities->rows!=dlogits->rows || probabilities->cols!=dlogits->cols){return 0;}
    
    for (int i=0; i<probabilities->rows; i++){
        int true_class=labels[i];
        for (int j=0; j<probabilities->cols; j++){
            double gradient=matrix_get (probabilities, i, j);
            if (j==true_class){gradient-=1.0;}
            gradient/=probabilities->rows;
            matrix_set(dlogits,i, j, gradient);
        }
    }
    return 1;
}

void loss_test(){
    Matrix logits;
    Matrix probabilities;
    Matrix dlogits;
    int labels[2];
    double loss;

    printf("\nLOSS TEST\n");
    logits=matrix_create(2, 3);
    matrix_set(&logits, 0, 0, 2.0);
    matrix_set(&logits, 0, 1, 1.0);
    matrix_set(&logits, 0, 2, 0.1);

    matrix_set(&logits, 1, 0, 0.5);
    matrix_set(&logits, 1, 1, 2.5);
    matrix_set(&logits, 1, 2, 0.3);

    labels[0]=0;
    labels[1]=1;
    probabilities=matrix_create(2, 3);
    dlogits=matrix_create(2, 3);

    softmax(&logits, &probabilities);
    loss=cross_entripy(&probabilities, labels);
    backward(&probabilities, labels, &dlogits);

    matrix_print(&logits, "logits");
    matrix_print(&probabilities, "probabilities");
    printf("cross entropy loss: %f\n", loss);
    matrix_print(&dlogits, "dlogits");

    matrix_free(&logits);
    matrix_free(&probabilities);
    matrix_free(&dlogits);
}