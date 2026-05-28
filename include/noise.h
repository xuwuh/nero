#ifndef NOISE_H
#define NOISE_H

#include "config.h"
#include "load_dataset.h"
#include "tensor.h"

//добавление шума
int dataset_add_noise(Dataset *dataset, double noise_ratio, double noise_value);

//добавление шума из конфига
int dataset_add_noise_from_config(Dataset *dataset, Config *config);

//добавление шума в тензор
int tensor_add_noise(Tensor *tensor, double noise_ratio, double noise_value);

#endif
