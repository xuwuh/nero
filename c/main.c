#include <stdio.h>
#include <stdlib.h>

#include "../include/config.h"
#include "../include/load_dataset.h"
#include "../include/matrix.h"
#include "../include/utils.h"
#include "../include/loss.h"
#include "../include/optimizer.h"
#include "../include/linear_layer.h"
#include "../include/tensor.h"
#include "../include/relu.h"
#include "../include/maxpool.h"
#include "../include/im2col.h"
#include "../include/conv.h"
#include "../include/flatten.h"

#include "../include/model.h"
#include "../include/train_cnn.h"
#include "../include/log.h"

int main(){
    Config config;
    Dataset train_dataset={0};
    Dataset test_dataset={0};

    if (!load("configs/config.txt", &config)){return 1;}; //загружаем конфиг

    if(!dataset_load(&train_dataset, config.train_data_path, &config, 100)){return 1;}
    if(!dataset_load(&test_dataset, config.test_data_path, &config, 100)){dataset_free(&train_dataset); return 1;}
   
    dataset_print(&train_dataset);
    dataset_print_labels(&train_dataset, 10);
    dataset_print_pixels(&train_dataset, 10);

    //тесты 
    conf_print(&config);

//    matrix_test();
//    utils_test();
//    loss_test();
//    optimizer_test();
//    linear_test();
//    tensor_tests();
//    relu_test();
//    maxpool_tests();
//    im2col_test();
//    col2im_test();
//    conv_tests();
//   conv_backward_tests();
//    flatten_test();
//    model_tests();

    //обучение
    srand(42);
    if (!train_cnn(&train_dataset, &test_dataset, &config, 0, "results/history_sgd.csv", "results/confusion_sgd.csv")){
        dataset_free(&test_dataset);
        dataset_free(&train_dataset); 
        return 1;}

    srand(42);
    if (!train_cnn(&train_dataset, &test_dataset, &config, 1, "results/history_momentum.csv", "results/confusion_momentum.csv")){
        dataset_free(&test_dataset);
        dataset_free(&train_dataset); 
        return 1;}

    dataset_free(&test_dataset);
    dataset_free(&train_dataset);

    return 0;
}
