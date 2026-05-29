#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
    int rows; //
    int cols; //
    double *data; //
} Matrix ;

Matrix matrix_create(int rows, int cols); //содание матрицы
void matrix_free(Matrix *matrix); //очищение памяти
void matrix_zero(Matrix *matrix); //заполнение нулями
void matrix_full_value(Matrix *matrix, double value); //заполнение одним знаением
double matrix_get(Matrix *matrix, int rosw, int cols); //получить значение
void matrix_set(Matrix *matrix, int rows, int cols, double value); //установить значение 
int matrix_mult(Matrix *a, Matrix *b, Matrix *res); //умнодение матриц
int matrix_transpose(Matrix *a, Matrix *res); //транспонирование
int matrix_copy(Matrix *a, Matrix *b); //копирование
int matrix_dias(Matrix *matrix, Matrix *bias); //добавление биаса

//вывод 
void matrix_print(Matrix *matrix, char *name);

#endif
