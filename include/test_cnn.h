#ifndef TEST_CNN_H
#define TEST_CNN_H

#include "config.h"
#include "load_dataset.h"
#include "model.h"

typedef struct
{
    double loss; //ошибка
    double accuracy; //точность
    int total_items; //количество объектов
} TestCNN;

TestCNN test_cnn(Model *model, Dataset *dataset, Config *config);
TestCNN test_cnn_with_confusion(Model *model, Dataset *dataset, Config *config, int *confusion_matrix);

#endif
