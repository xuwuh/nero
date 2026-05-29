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
    int out_h=maxpool_output_size(input->h, layer->pool_size, layer->stride);
    int out_w=maxpool_output_size(input->w, layer->pool_size, layer->stride);
    if (output->n!=input->n || output->c!=input->c || output->h!=out_h || output->w!=out_w){return 0;}

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
    int out_h=maxpool_output_size(dinput->h, layer->pool_size, layer->stride);
    int out_w=maxpool_output_size(dinput->w, layer->pool_size, layer->stride);
    if (doutput->n!=dinput->n || doutput->c!=dinput->c || doutput->h!=out_h || doutput->w!=out_w){return 0;}
    
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
