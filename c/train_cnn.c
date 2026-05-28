#include <stdio.h>
#include <stdlib.h>

#include "../include/train_cnn.h"
#include "../include/tensor.h"
#include "../include/matrix.h"
#include "../include/loss.h"
#include "../include/utils.h"
#include "../include/test_cnn.h"
#include "../include/log.h"
#include "../include/noise.h"

//возвращаем индекс максимальной вероятности из матрицы: probabilities:batch_size*num_classes || row: номер объекта в batch
int matrix_row_argmax(Matrix *matrix, int row){
    return max_arg(&matrix->data[row*matrix->cols],matrix->cols);
}

//считает accuracy=correct/batch_size
double cnn_batch_accuracy(Matrix *probabilities,int *labels){
    int correct=0;
    for (int i=0; i<probabilities->rows; i++){
        int predicted_class=matrix_row_argmax(probabilities, i);
        if (predicted_class==labels[i]){correct++;}
    }
    return (double)correct/(double)probabilities->rows;
}

//копируем батч из датасета сразу в тензор: batch*channels*height*width
int dataset_get_tensor_batch(Dataset *dataset, int start_index, int batch_size, Tensor *batch_images, int *batch_labels){
    int actual_batch_size=batch_size;
    if (start_index+actual_batch_size>dataset->image_count){actual_batch_size=dataset->image_count-start_index;}

    //копируем каждое изображен ие в тензор
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

//обучение 
int train_cnn(Dataset *train_dataset, Dataset *test_dataset, Config *config, int use_momentum, char *history_file){
    Model model;

    if (train_dataset==NULL || config==NULL){printf("Error: train_cnn got NULL argument\n");return 0;}
    if (train_dataset->image_count<=0){printf("Error: empty train dataset\n");return 0;}
    if (!model_init(&model, config)){printf("Error: failed to initialize CNN model\n");return 0;}
    if (history_file!=NULL){log_init_history(history_file);}

    printf("TRAINING START\n");
    if (use_momentum){printf("SGD+Momentum\n");
    }
    else{printf("SGD\n");}

    double train_start_time=check_time();

    for (int epoch=0; epoch<config->epochs; epoch++){
        int batch_count=0;
        double epoch_loss_sum=0.0;
        double epoch_accuracy_sum=0.0;
        double epoch_start_time=check_time();
        double test_loss=0.0;
        double test_accuracy=0.0;

        //проходимся батчами по датасету
        for (int start=0; start<train_dataset->image_count; start+=config->batch_size){
            int actual_batch_size=config->batch_size;

            if (start+actual_batch_size>train_dataset->image_count){
                actual_batch_size=train_dataset->image_count-start;
            }

            //выделяем класс из батча
            int *batch_labels=(int *)safe_malloc(sizeof(int)*actual_batch_size);

            //batch*channels*height*width
            Tensor batch_input=tensor_create(actual_batch_size, train_dataset->image_channels, train_dataset->image_height, train_dataset->image_width);

            //буферы для прямой проходки 
            Tensor conv_out=tensor_create(actual_batch_size, config->conv_out_channels, model.conv_out_h, model.conv_out_w);
            Tensor relu_out=tensor_create(actual_batch_size, config->conv_out_channels, model.conv_out_h, model.conv_out_w);
            Tensor pool_out=tensor_create(actual_batch_size, config->conv_out_channels, model.pool_out_h, model.pool_out_w);
            Matrix flat_out=matrix_create(actual_batch_size, model.flatten_size);
            Matrix logits=matrix_create(actual_batch_size, config->classes);
            Matrix probabilities=matrix_create(actual_batch_size, config->classes);
            Matrix dlogits=matrix_create(actual_batch_size, config->classes);

            //копируем батч из датасета в тензор
            dataset_get_tensor_batch(train_dataset, start, actual_batch_size, &batch_input, batch_labels);
            if (config->use_noise && !tensor_add_noise(&batch_input, config->noise_ratio, config->noise_value)){
                free(batch_labels);
                tensor_free(&batch_input);
                tensor_free(&conv_out);
                tensor_free(&relu_out);
                tensor_free(&pool_out);
                matrix_free(&flat_out);
                matrix_free(&logits);
                matrix_free(&probabilities);
                matrix_free(&dlogits);
                model_free(&model);
                return 0;
            }

            //прямая проходка
            model_forward(&model, &batch_input, &conv_out, &relu_out, &pool_out, &flat_out, &logits, &probabilities);

            //считаем ошибку и вероятность правильных ответов
            double loss=cross_entripy(&probabilities, batch_labels);
            double accuracy=cnn_batch_accuracy(&probabilities, batch_labels);
            epoch_loss_sum+=loss;
            epoch_accuracy_sum+=accuracy;

            //обратная проходка
            backward(&probabilities, batch_labels, &dlogits);
            model_backward(&model, &batch_input, &conv_out, &relu_out, &pool_out, &flat_out, &dlogits);

            //обновляем веса
            if (use_momentum){
                model_momentum(&model, config->learning_rate, config->momentum, config->regularizator);
            }
            else{
                model_sgd(&model, config->learning_rate, config->regularizator);
            }
            batch_count++;

            //освобождаем память
            free(batch_labels);
            tensor_free(&batch_input);
            tensor_free(&conv_out);
            tensor_free(&relu_out);
            tensor_free(&pool_out);
            matrix_free(&flat_out);
            matrix_free(&logits);
            matrix_free(&probabilities);
            matrix_free(&dlogits);
        }

        double epoch_end_time=check_time();

        //выводим средние значения по эпохе
        if (batch_count>0){
            double avg_loss=epoch_loss_sum/batch_count;
            double avg_accuracy=epoch_accuracy_sum/batch_count;
            double epoch_time=epoch_end_time-epoch_start_time;
            printf("epoch %d/%d | loss: %.6f | accuracy: %.4f | time: %.3f sec\n", epoch+1, config->epochs, avg_loss, avg_accuracy, epoch_time);

            if (test_dataset!=NULL){
                TestCNN test_result;
                test_result=test_cnn(&model, test_dataset, config);
                test_loss=test_result.loss;
                test_accuracy=test_result.accuracy;
                printf("test_loss: %.6f | test_accuracy: %.4f | test_items: %d\n", test_result.loss, test_result.accuracy, test_result.total_items);
            }
            if (history_file!=NULL){
                log_log_epoch(history_file, epoch+1, avg_loss, avg_accuracy, test_loss, test_accuracy, epoch_time);
            }
        }
    }

    double train_end_time=check_time();
    printf("total training time: %.3f sec\n",train_end_time-train_start_time);

    model_free(&model);
    return 1;
}
