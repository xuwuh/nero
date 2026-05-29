#include <stdio.h>

#include "model_save.h"

//запись матрицы в json
static void model_save_matrix(FILE *file, char *name, Matrix *matrix, int last){
    int total=matrix->rows*matrix->cols;

    fprintf(file, "    \"%s\": {\"rows\": %d, \"cols\": %d, \"data\": [", name, matrix->rows, matrix->cols);
    for (int i=0; i<total; i++){
        fprintf(file, "%.17g", matrix->data[i]);
        if (i<total-1){fprintf(file, ", ");}
    }
    fprintf(file, "]}");
    if (!last){fprintf(file, ",");}
    fprintf(file, "\n");
}

//сохранение весов модели
int model_save(Model *model, Config *config, char *filename){
    FILE *file;

    if (model==NULL || config==NULL || filename==NULL){return 0;}

    file=fopen(filename, "w");
    if (file==NULL){printf("Error: failed to open model weights file\n");return 0;}

    fprintf(file, "{\n");
    fprintf(file, "  \"input_w\": %d,\n", config->input_w);
    fprintf(file, "  \"input_h\": %d,\n", config->input_h);
    fprintf(file, "  \"input_ch\": %d,\n", config->input_ch);
    fprintf(file, "  \"classes\": [");
    for (int i=0; i<config->classes; i++){
        fprintf(file, "\"%c\"", 'A'+i);
        if (i<config->classes-1){fprintf(file, ", ");}
    }
    fprintf(file, "],\n");

    fprintf(file, "  \"conv\": {\n");
    fprintf(file, "    \"out_channels\": %d,\n", config->conv_out_channels);
    fprintf(file, "    \"kernel_h\": %d,\n", config->conv_kernel_h);
    fprintf(file, "    \"kernel_w\": %d,\n", config->conv_kernel_w);
    fprintf(file, "    \"stride\": %d,\n", config->conv_stride);
    fprintf(file, "    \"padding\": %d,\n", config->conv_padding);
    model_save_matrix(file, "weights", &model->conv.weights, 0);
    model_save_matrix(file, "bias", &model->conv.bias, 1);
    fprintf(file, "  },\n");

    fprintf(file, "  \"pool\": {\n");
    fprintf(file, "    \"size\": %d,\n", config->pool_size);
    fprintf(file, "    \"stride\": %d\n", config->pool_stride);
    fprintf(file, "  },\n");

    fprintf(file, "  \"linear\": {\n");
    model_save_matrix(file, "weights", &model->final_layer.weights, 0);
    model_save_matrix(file, "bias", &model->final_layer.bias, 1);
    fprintf(file, "  }\n");
    fprintf(file, "}\n");

    fclose(file);
    return 1;
}
