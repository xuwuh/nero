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
    int expendet_row=output->n*out_h*out_w;
    int expendet_col=output->c*kernel_h*kernel_w;
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

//тесты
void im2col_test(){
    Tensor input;
    Matrix col;
    int kernel_h;
    int kernel_w;
    int stride;
    int padding;
    int out_h;
    int out_w;

    printf("\nIM2COL TEST\n");

    input=tensor_create(1, 1, 3, 3);
    tensor_set(&input, 0, 0, 0, 0, 1.0);
    tensor_set(&input, 0, 0, 0, 1, 2.0);
    tensor_set(&input, 0, 0, 0, 2, 3.0);
    tensor_set(&input, 0, 0, 1, 0, 4.0);
    tensor_set(&input, 0, 0, 1, 1, 5.0);
    tensor_set(&input, 0, 0, 1, 2, 6.0);
    tensor_set(&input, 0, 0, 2, 0, 7.0);
    tensor_set(&input, 0, 0, 2, 1, 8.0);
    tensor_set(&input, 0, 0, 2, 2, 9.0);

    kernel_h=2;
    kernel_w=2;
    stride=1;
    padding=0;
    out_h=conv_output_size(input.h, kernel_h, stride, padding);
    out_w=conv_output_size(input.w, kernel_w, stride, padding);

    col=matrix_create(input.n * out_h * out_w, input.c * kernel_h * kernel_w);
    im2col(&input, &col, kernel_h, kernel_w, stride, padding);
    tensor_print(&input, "im2col input");
    matrix_print(&col, "im2col result");

    tensor_free(&input);
    matrix_free(&col);
}

void col2im_test(){
    Matrix col;
    Tensor output;

    printf("\nCOL2IM TEST\n");

    col=matrix_create(4, 4);
    output=tensor_create(1, 1, 3, 3);

    matrix_full_value(&col, 1.0);
    col2im(&col, &output, 2, 2, 1, 0);
    tensor_print(&output, "col2im output");

    printf("col2im output values:\n");
    printf("%.2f %.2f %.2f\n",tensor_get(&output, 0, 0, 0, 0),tensor_get(&output, 0, 0, 0, 1),tensor_get(&output, 0, 0, 0, 2));
    printf("%.2f %.2f %.2f\n",tensor_get(&output, 0, 0, 1, 0),tensor_get(&output, 0, 0, 1, 1),tensor_get(&output, 0, 0, 1, 2));
    printf("%.2f %.2f %.2f\n",tensor_get(&output, 0, 0, 2, 0),tensor_get(&output, 0, 0, 2, 1),tensor_get(&output, 0, 0, 2, 2));

    matrix_free(&col);
    tensor_free(&output);
}