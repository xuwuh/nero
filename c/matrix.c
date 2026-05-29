#include <stdio.h>
#include <stdlib.h>

#include "../include/matrix.h"

void *matrix_mallok(size_t size){
    void *ptr=malloc(size);
    return ptr;
}

//содание матрицы
Matrix matrix_create(int rows, int cols){
    Matrix matrix;
    matrix.rows=rows;
    matrix.cols=cols;
    matrix.data=(double *)matrix_mallok(sizeof(double)*rows*cols);
    matrix_zero(&matrix);
    return matrix;
}

//очищение памяти
void matrix_free(Matrix *matrix){
    if (matrix==NULL){return;}
    if (matrix->data!=NULL){free(matrix->data); matrix->data=NULL;}
    matrix->rows=0;
    matrix->cols=0;
}

//заполнение нулями
void matrix_zero(Matrix *matrix){
    int total=matrix->rows*matrix->cols;
    for (int i=0; i<total; i++){
        matrix->data[i]=0.0;
    }
}

//заполнение одним знаением
void matrix_full_value(Matrix *matrix, double value){
    int total=matrix->rows*matrix->cols;
    for (int i=0; i<total; i++){
        matrix->data[i]=value;
    }
}

//получить значение
double matrix_get(Matrix *matrix, int row, int col){
    return matrix->data[row*matrix->cols+col];
}

//установить значение 
void matrix_set(Matrix *matrix, int row, int col, double value){
    matrix->data[row*matrix->cols+col]=value;
}

//умнодение матриц: C[i][j]=summ(A[i][k]*B[k][j])
int matrix_mult(Matrix *a, Matrix *b, Matrix *res){
    if (a->cols != b->rows){printf("ERROR: a->cols != b->rows"); return 0;}
    if (res->rows != a->rows || res->cols != b->cols){printf("ERROR res->rows != a->rows || res->cols != b->cols"); return 0;}

    matrix_zero(res);
    for (int i=0; i<a->rows; i++){
        for (int j=0; j<b->cols; j++){
            double sum=0.0;
            for (int k=0; k<a->cols; k++){
                sum+=matrix_get(a,i,k)*matrix_get(b,k,j);
            }
            matrix_set(res, i, j, sum);
        }
    }
    return 1;
}

//транспонирование: res[i][j]=a[j][i]
int matrix_transpose(Matrix *a, Matrix *res){
    if (res->rows!=a->cols || res->cols!=a->rows){return 0;}
    for (int i=0; i<a->rows; i++){
        for (int j=0; j<a->cols; j++){
            matrix_set (res, j, i, matrix_get(a, i, j));
        }
    }
    return 1;
}

//копирование
int matrix_copy(Matrix *a, Matrix *b){
    if (a==NULL || b==NULL || a->rows!=b->rows || a->cols!=b->cols){return 0;}
    int total=a->rows*a->cols;
    for (int i=0; i<total; i++){
        b->data[i]=a->data[i];
    }
    return 1;
} 

//добавление биаса
int matrix_dias(Matrix *matrix, Matrix *bias){
    if (bias->rows!=1 || bias->cols!=matrix->cols){return 0;}
    for (int i=0; i<matrix->rows; i++){
        for (int j=0; j<matrix->cols; j++){
            double value;
            value=matrix_get(matrix, i, j);
            value+=matrix_get(bias, 0, j);
            matrix_set(matrix, i, j, value);
        }
    }
    return 1;
} 

//вывод
void matrix_print(Matrix *matrix, char *name){
    printf("%s [%d x %d]: \n", name, matrix->rows, matrix->cols);
    for (int i=0; i<matrix->rows; i++){
        for (int j=0; j<matrix->cols; j++){
            printf("%f ", matrix_get(matrix, i, j));
        }
        printf("\n");
    }
    printf("\n");
}
