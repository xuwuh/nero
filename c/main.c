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

int main(){
    Config config;
    Dataset train_dataset;
    Optimizer optimizer;
    LinearLayer output_layer;

    if (!load("configs/config.txt", &config)){return 1;}; //загружаем конфиг

    optimizer=optimizer_create(config.learning_rate, config.momentum, config.regularizator);
    int input_size=config.input_w*config.input_h*config.input_ch;

    if(!dataset_load(&train_dataset, config.train_data_path, &config, 100)){return 1;}
    if (!linear_init(&output_layer, input_size, config.classes)){dataset_free(&train_dataset);return 1;}
   
    dataset_print(&train_dataset);
    dataset_print_labels(&train_dataset, 10);
    dataset_print_pixels(&train_dataset, 10);

    //тесты 
    conf_print(&config);

    matrix_test();
    utils_test();
    loss_test();
    optimizer_test();
    linear_test();
    tensor_tests();
    relu_test();

    linear_free(&output_layer);
    dataset_free(&train_dataset);

    return 0;
}