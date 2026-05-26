#include <stdio.h>

#include "../include/maxpool.h"

//инициализация слоя
int maxpool_init(MaxPool *layer, int pool_size, int stride){
    layer->pool_size =pool_size;
    layer->stride=stride;
    return 1;
}

//размер выхода после слоя: out = (input_size - pool_size) / stride + 1
int maxpool_output_size(int input_size, int pool_size, int stride){
    return (input_size-pool_size)/stride+1;
}

//прямой проход
int maxpool_forward(MaxPool *layer, Tensor *input, Tensor *output){
    for (int n=0; n<input->n; n++){
        for (int c=0; c<input->c; c++){
            for (int h=0; h<output->h; h++){
                for (int w=0; w<output->w; w++){
                    int input_h_start=h*layer->stride;
                    int input_w_start=w*layer->stride;
                    double max_value=0.0;
                    int first_value=1;

                    for (int ph=0; ph<layer->pool_size ; ph++){
                        for (int pw=0; pw<layer->pool_size ; pw++){
                            int ih=input_h_start+ph;
                            int iw=input_w_start+pw;
                            double value=tensor_get(input, n, c, ih, iw);

                            if (first_value || value>max_value){
                                max_value=value;
                                first_value=0;
                            }
                        }
                    }
                    tensor_set(output, n,c , h, w, max_value);
                }
            }
        }
    }
    return 1;
}

//обратный проход
int maxpool_backward(MaxPool *layer, Tensor *input, Tensor *doutput, Tensor *dinput){
    tensor_zero(dinput);
    for (int n=0; n<input->n; n++){
        for (int c=0; c<input->c; c++){
            for (int h=0; h<doutput->h; h++){
                for (int w=0; w<doutput->w; w++){
                    int input_h_start=h*layer->stride;
                    int input_w_start=w*layer->stride;
                    int max_h=input_h_start;
                    int max_w=input_w_start;
                    double max_value=0.0;
                    int first_value=1;

                    for (int ph=0; ph<layer->pool_size ; ph++){
                        for (int pw=0; pw<layer->pool_size ; pw++){
                            int ih=input_h_start+ph;
                            int iw=input_w_start+pw;
                            double value=tensor_get(input, n, c, ih, iw);

                            if (first_value || value>max_value){
                                max_value=value;
                                max_h=ih;
                                max_w=iw;
                                first_value=0;
                            }
                        }
                    }
                    double grad=tensor_get(doutput, n, c, h, w);
                    double current=tensor_get(dinput, n, c, max_h, max_w);
                    tensor_set(dinput, n, c, max_h, max_w, grad+current);
                }
            }
        }
    }
    return 1;
}

//тесты
void maxpool_tests(void){
    MaxPool pool;
    Tensor input;
    Tensor output;
    Tensor doutput;
    Tensor dinput;

    printf("\nMAXPOOL TEST\n");

    maxpool_init(&pool, 2, 2);
    input=tensor_create(1, 1, 4, 4);
    output=tensor_create(1, 1, 2, 2);
    doutput=tensor_create(1, 1, 2, 2);
    dinput=tensor_create(1, 1, 4, 4);


    tensor_set(&input, 0, 0, 0, 0, 1.0);
    tensor_set(&input, 0, 0, 0, 1, 2.0);
    tensor_set(&input, 0, 0, 0, 2, 5.0);
    tensor_set(&input, 0, 0, 0, 3, 4.0);
    tensor_set(&input, 0, 0, 1, 0, 3.0);
    tensor_set(&input, 0, 0, 1, 1, 6.0);
    tensor_set(&input, 0, 0, 1, 2, 7.0);
    tensor_set(&input, 0, 0, 1, 3, 8.0);
    tensor_set(&input, 0, 0, 2, 0, 1.0);
    tensor_set(&input, 0, 0, 2, 1, 2.0);
    tensor_set(&input, 0, 0, 2, 2, 9.0);
    tensor_set(&input, 0, 0, 2, 3, 0.0);
    tensor_set(&input, 0, 0, 3, 0, 4.0);
    tensor_set(&input, 0, 0, 3, 1, 3.0);
    tensor_set(&input, 0, 0, 3, 2, 2.0);
    tensor_set(&input, 0, 0, 3, 3, 1.0);

    maxpool_forward(&pool, &input, &output);

    tensor_set(&doutput, 0, 0, 0, 0, 1.0);
    tensor_set(&doutput, 0, 0, 0, 1, 2.0);
    tensor_set(&doutput, 0, 0, 1, 0, 3.0);
    tensor_set(&doutput, 0, 0, 1, 1, 4.0);

    maxpool_backward(&pool, &input, &doutput, &dinput);

    tensor_print(&input, "MaxPool input");
    tensor_print(&output, "MaxPool output");

    printf("MaxPool output values:\n");
    printf("%.2f %.2f\n", tensor_get(&output, 0, 0, 0, 0), tensor_get(&output, 0, 0, 0, 1));
    printf("%.2f %.2f\n", tensor_get(&output, 0, 0, 1, 0), tensor_get(&output, 0, 0, 1, 1));

    printf("\nMaxPool dinput values:\n");
    printf("%.2f %.2f %.2f %.2f\n", tensor_get(&dinput, 0, 0, 0, 0), tensor_get(&dinput, 0, 0, 0, 1), tensor_get(&dinput, 0, 0, 0, 2), tensor_get(&dinput, 0, 0, 0, 3));
    printf("%.2f %.2f %.2f %.2f\n", tensor_get(&dinput, 0, 0, 1, 0), tensor_get(&dinput, 0, 0, 1, 1), tensor_get(&dinput, 0, 0, 1, 2), tensor_get(&dinput, 0, 0, 1, 3));
    printf("%.2f %.2f %.2f %.2f\n", tensor_get(&dinput, 0, 0, 2, 0), tensor_get(&dinput, 0, 0, 2, 1), tensor_get(&dinput, 0, 0, 2, 2), tensor_get(&dinput, 0, 0, 2, 3));
    printf("%.2f %.2f %.2f %.2f\n", tensor_get(&dinput, 0, 0, 3, 0), tensor_get(&dinput, 0, 0, 3, 1), tensor_get(&dinput, 0, 0, 3, 2), tensor_get(&dinput, 0, 0, 3, 3));

    tensor_free(&input);
    tensor_free(&output);
    tensor_free(&doutput);
    tensor_free(&dinput);

    printf("\nMaxPool tests finished.\n");
}