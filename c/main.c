#include <stdio.h>

#include "../include/config.h"
#include "../include/load_dataset.h"

int main(){
    Config config;
    Dataset train_dataset={0};

    load("configs/config.txt", &config); //загружаем конфиг

    dataset_load(&train_dataset, config.train_data_path, &config, 100);
   
    dataset_print(&train_dataset);
    dataset_print_labels(&train_dataset, 10);
    dataset_print_pixels(&train_dataset, 10);

    //тесты 
    conf_print(&config);

    dataset_free(&train_dataset);

    return 0;
}