#ifndef LOAD_DATASET_H
#define LOAD_DATASET_H

#include "config.h"

typedef struct
{
    double *images; //все изображения
    int *lable; //Метки классов 
    int image_count; //кол-во изображений
    int image_size; //размер изображений в пикселях
    int image_width; //ширина
    int image_height; //высота
    int image_channels; //каналы
    int class_num; //классы
} Dataset;

//загружаем датасет 
int dataset_load(
    Dataset *dataset,
    char *filename,
    Config *config,
    int max_images
);

//осваобождаем память
void dataset_free(Dataset *dataset);

//печатаем информацию для отладки 
void dataset_print(Dataset *dataset);

//печатаем несколько меток и пиксели (проверяем нормализацию)
void dataset_print_labels(Dataset *dataset, int n);
void dataset_print_pixels(Dataset *dataset, int n);

#endif

