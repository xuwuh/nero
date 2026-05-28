#include <stdio.h>
#include <stdlib.h>

#include "../include/test_cnn.h"
#include "../include/tensor.h"
#include "../include/matrix.h"
#include "../include/loss.h"
#include "../include/utils.h"

//индекс максимума в строке
static int test_row_argmax(Matrix *matrix, int row){
    return max_arg(&matrix->data[row*matrix->cols],matrix->cols);
}

//точность батчей
static double test_batch_accuracy(Matrix *probabilities, int *labels){
    int correct=0;
    for (int i=0; i<probabilities->rows; i++){
        int predicted_class=test_row_argmax(probabilities, i);
        if (predicted_class==labels[i]){correct++;}
    }
    return (double)correct/(double)probabilities->rows;
}

//копирование батчей
static int test_get_tensor_batch(Dataset *dataset, int start_index, int batch_size, Tensor *batch_images, int *batch_labels){
    int actual_batch_size=batch_size;
    if (start_index+actual_batch_size>dataset->image_count){actual_batch_size=dataset->image_count-start_index;}

    for (int i=0; i<actual_batch_size; i++){
        int dataset_index=start_index+i;
        batch_labels[i]=dataset->lable[dataset_index];
        for (int c=0; c<dataset->image_channels; c++){
            for (int h=0; h<dataset->image_height; h++){
                for (int w=0; w<dataset->image_width; w++){
                    int flat_index=c*dataset->image_height*dataset->image_width+h*dataset->image_width+w;
                    double pixel=dataset->images[dataset_index*dataset->image_size+flat_index];
                    tensor_set(batch_images, i, c, h, w, pixel);
                }
            }
        }
    }
    return actual_batch_size;
}

TestCNN test_cnn(Model *model, Dataset *dataset, Config *config){
    TestCNN result={0};
    double loss_sum=0.0;
    double accuracy_sum=0.0;
    int batch_count=0;

    if (model==NULL || dataset==NULL || config==NULL){printf("Error: test_cnn got NULL argument\n");return result;}
    if (dataset->image_count<=0){printf("Error: test dataset is empty\n");return result;}

    for (int start=0; start<dataset->image_count; start+=config->batch_size){
        int actual_batch_size=config->batch_size;
        if (start+actual_batch_size>dataset->image_count){actual_batch_size=dataset->image_count-start;}

        int *batch_labels=(int *)safe_malloc(sizeof(int)*actual_batch_size);
        Tensor batch_input=tensor_create(actual_batch_size,dataset->image_channels,dataset->image_height,dataset->image_width);
        Tensor conv_out=tensor_create(actual_batch_size,config->conv_out_channels,model->conv_out_h,model->conv_out_w);
        Tensor relu_out=tensor_create(actual_batch_size,config->conv_out_channels,model->conv_out_h,model->conv_out_w);
        Tensor pool_out=tensor_create(actual_batch_size,config->conv_out_channels,model->pool_out_h,model->pool_out_w);
        Matrix flat_out=matrix_create(actual_batch_size,model->flatten_size);
        Matrix logits=matrix_create(actual_batch_size,config->classes);
        Matrix probabilities=matrix_create(actual_batch_size,config->classes);

        if (!test_get_tensor_batch(dataset, start, actual_batch_size, &batch_input, batch_labels)){
            free(batch_labels);
            tensor_free(&batch_input);
            tensor_free(&conv_out);
            tensor_free(&relu_out);
            tensor_free(&pool_out);
            matrix_free(&flat_out);
            matrix_free(&logits);
            matrix_free(&probabilities);
            return result;
        }

        if (!model_forward(model, &batch_input, &conv_out, &relu_out, &pool_out, &flat_out, &logits, &probabilities)){
            printf("Error: test model_forward failed\n");
            free(batch_labels);
            tensor_free(&batch_input);
            tensor_free(&conv_out);
            tensor_free(&relu_out);
            tensor_free(&pool_out);
            matrix_free(&flat_out);
            matrix_free(&logits);
            matrix_free(&probabilities);
            return result;
        }

        loss_sum+=cross_entripy(&probabilities, batch_labels);
        accuracy_sum+=test_batch_accuracy(&probabilities, batch_labels);
        batch_count++;
        result.total_items+=actual_batch_size;

        free(batch_labels);
        tensor_free(&batch_input);
        tensor_free(&conv_out);
        tensor_free(&relu_out);
        tensor_free(&pool_out);
        matrix_free(&flat_out);
        matrix_free(&logits);
        matrix_free(&probabilities);
    }

    if (batch_count>0){
        result.loss=loss_sum/batch_count;
        result.accuracy=accuracy_sum/batch_count;
    }
    return result;
}
