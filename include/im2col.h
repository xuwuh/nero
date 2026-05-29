#ifndef IM2COL_H
#define IM2COL_H

#include "tensor.h"
#include "matrix.h"

//подсчет выходного значения после свертки
int conv_output_size(int input_size, int kernel_size, int stride, int padding);
//свертывание из тензора в матрицу
int im2col(Tensor *input, Matrix *col, int kernel_h, int kernel_w, int stride, int padding);
//отбраная функиця, развертка из матрицы в тензор 
int col2im(Matrix *col, Tensor *output, int kernel_h, int kernel_w, int stride, int padding);

//тесты
#endif
