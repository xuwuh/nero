#ifndef MODEL_SAVE_H
#define MODEL_SAVE_H

#include "../include/config.h"
#include "../include/model.h"

//сохранение весов модели
int model_save(Model *model, Config *config, char *filename);

#endif
