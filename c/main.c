#include <stdio.h>

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

int main(){
    Config config;
    Dataset train_dataset={0};

    if (!load("configs/config.txt", &config)){return 1;}; //загружаем конфиг

    if(!dataset_load(&train_dataset, config.train_data_path, &config, 100)){return 1;}
   
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
    if (!train_cnn(&train_dataset, &config, config.momentum > 0.0)){
        dataset_free(&train_dataset); 
        return 1;}

    dataset_free(&train_dataset);

    return 0;
}
