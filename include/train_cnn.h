#ifndef TRAIN_CNN_H
#define TRAIN_CNN_H

#include "config.h"
#include "load_dataset.h"
#include "model.h"

int train_cnn(Dataset *train_dataset, Config *config, int use_momentum);

#endif