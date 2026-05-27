#ifndef CONFIG_H
#define CONFIG_H

typedef struct
{
    int input_w;
    int input_h;
    int input_ch;
    int classes;

    int conv_out_channels;
    int conv_kernel_h;
    int conv_kernel_w;
    int conv_stride;
    int conv_padding;   

    int pool_size;
    int pool_stride;

    int epochs;
    int batch_size;
    double momentum;
    double learning_rate;
    double regularizator;

    char train_data_path[256];
    char test_data_path[256];
} Config;

int load(char *filename, Config *config);
int validate(Config *config);
void conf_print(Config *config);

#endif