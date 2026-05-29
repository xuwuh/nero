#include <stdio.h>

#include "../include/im2col.h"

//подсчет выходного значения после свертки: out = (input_size - kernel_size + 2 * padding) / stride + 1
int conv_output_size(int input_size, int kernel_size, int stride, int padding){
    return (input_size-kernel_size+2*padding)/stride+1;
}

//свертывание из тензора в матрицу
int im2col(Tensor *input, Matrix *col, int kernel_h, int kernel_w, int stride, int padding){
    int out_h=conv_output_size(input->h, kernel_h, stride, padding);
    int out_w=conv_output_size(input->w, kernel_w, stride, padding);
    int expected_rows=input->n*out_h*out_w;
    int expected_cols=input->c*kernel_h*kernel_w;
    if (col==NULL || col->rows!=expected_rows || col->cols!=expected_cols){return 0;}
    matrix_zero(col);

    for (int n=0; n<input->n; n++){
        for (int out_y=0; out_y<out_h; out_y++){
            for (int out_x=0; out_x<out_w; out_x++){
                int row_i=n*out_h*out_w+out_y*out_w+out_x;

                for (int c=0; c<input->c; c++){
                    for (int kernel_y=0; kernel_y<kernel_h; kernel_y++){
                        for (int kernel_x=0; kernel_x<kernel_w; kernel_x++){
                            int input_y=out_y*stride+kernel_y-padding;
                            int input_x=out_x*stride+kernel_x-padding;

                            int col_i=c*kernel_h*kernel_w+kernel_y*kernel_w+kernel_x;

                            double value;
                            if (input_y<0 || input_y>=input->h || input_x<0 || input_x>=input->w){value=0.0;}
                            else {value=tensor_get(input, n, c, input_y, input_x);}
                            matrix_set(col, row_i, col_i, value);
                        }
                    }
                }
            }
        }
    }
    return 1;
}

//отбраная функиця, развертка из матрицы в тензор 
int col2im(Matrix *col, Tensor *output, int kernel_h, int kernel_w, int stride, int padding){
    int out_h=conv_output_size(output->h, kernel_h, stride, padding);
    int out_w=conv_output_size(output->w, kernel_w, stride, padding);
    int expected_rows=output->n*out_h*out_w;
    int expected_cols=output->c*kernel_h*kernel_w;
    if (col==NULL || col->rows!=expected_rows || col->cols!=expected_cols){return 0;}
    tensor_zero(output);

    for (int n=0; n<output->n; n++){
        for (int out_y=0; out_y<out_h; out_y++){
            for (int out_x=0; out_x<out_w; out_x++){
                int row_i=n*out_h*out_w+out_y*out_w+out_x;

                for (int c=0; c<output->c; c++){
                    for (int kernel_y=0; kernel_y<kernel_h; kernel_y++){
                        for (int kernel_x=0; kernel_x<kernel_w; kernel_x++){
                            int input_y=out_y*stride+kernel_y-padding;
                            int input_x=out_x*stride+kernel_x-padding;

                            if (input_y<0 || input_y>=output->h || input_x<0 || input_x>=output->w){continue;}

                            int col_i=c*kernel_h*kernel_w+kernel_y*kernel_w+kernel_x;

                            double old_val=tensor_get(output, n, c, input_y, input_x);
                            double add_val=matrix_get(col, row_i, col_i);
                            tensor_set(output, n, c, input_y, input_x, old_val+add_val);
                        }
                    }
                }
            }
        }
    }
    return 1;
}

