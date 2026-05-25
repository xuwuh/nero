#include <stdio.h>

#include "../include/config.h"
#include "../include/load_dataset.h"
#include "../include/matrix.h"
#include "../include/utils.h"
#include "../include/loss.h"
#include "../include/optimizer.h"

int main(){
    Config config;
    Dataset train_dataset;
    Optimizer optimizer;

    load("configs/config.txt", &config); //загружаем конфиг

    optimizer=optimizer_create(config.learning_rate, config.momentum, config.regularizator);

    dataset_load(&train_dataset, config.train_data_path, &config, 100);
   
    dataset_print(&train_dataset);
    dataset_print_labels(&train_dataset, 10);
    dataset_print_pixels(&train_dataset, 10);

    //тесты 
    conf_print(&config);

    matrix_test();
    utils_test();
    loss_test();
    optimizer_test();

    dataset_free(&train_dataset);

    return 0;
}