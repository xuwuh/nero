#include <stdio.h>
#include <stdlib.h>

#include "../include/noise.h"
#include "../include/utils.h"

//добавление шума
int dataset_add_noise(Dataset *dataset, double noise_ratio, double noise_value){
    int pixel_count=dataset->image_count*dataset->image_size;
    int noise_count=(int)(pixel_count*noise_ratio);
    int changed_count=0;

    for (int i=0; i<noise_count; i++){
        int pixel_i=rand()%pixel_count;
        int sign=rand()%2;
        double old_value=dataset->images[pixel_i];
        double new_value;

        if (sign==0){new_value=old_value-noise_value;}
        else{new_value=old_value+noise_value;}

        dataset->images[pixel_i]=clamp_double(new_value, 0.0, 1.0);
        changed_count++;
    }

    printf("Noise added: changed %d pixels from %d total pixels, ratio %.2f, value %.2f\n", changed_count, pixel_count, noise_ratio, noise_value);
    return 1;
}

//добавление шума из конфига
int dataset_add_noise_from_config(Dataset *dataset, Config *config){
    return dataset_add_noise(dataset, config->noise_ratio, config->noise_value);
}

//добавление шума в тензор
int tensor_add_noise(Tensor *tensor, double noise_ratio, double noise_value){
    int noise_count=(int)(tensor->size*noise_ratio);
    for (int i=0; i<noise_count; i++){
        int pixel_i=rand()%tensor->size;
        int sign=rand()%2;
        double old_value=tensor->data[pixel_i];
        double new_value;

        if (sign==0){new_value=old_value-noise_value;}
        else{new_value=old_value+noise_value;}

        tensor->data[pixel_i]=clamp_double(new_value, 0.0, 1.0);
    }
    return 1;
}
