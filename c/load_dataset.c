#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/load_dataset.h"
#include "../include/utils.h"

//нормализация пикселей
double normal_score(double value, double min_value, double max_value){
    if (value<min_value){return min_value;}
    if (value>max_value){return max_value;}
    return value;
}
double normal (int pixel){
    double norm;
    norm=(double)pixel/255.0;
    return normal_score(norm, 0.0, 1.0);
}

//считаем сколько у нас всего строк в CSV файле, нужно для выделения памяти
int count_line_file(char *filenmae){
    FILE *file;
    int line=0, ch;
    file=fopen(filenmae, "r");
    if (file==NULL){printf("ERROR: caccon open dataset file\n"); return -1;}
    while ((ch=fgetc(file))!=EOF){
        if (ch=='\n'){line++;}
    }
    fclose(file);
    return line;
}

//читаем даатсет
int dataset_load(Dataset *dataset, char *filename, Config *config, int max_images){
    FILE *file;
    char line[20000];
    int total_line; 
    int pixel;
    int capacity;
    int image_index;

    if (dataset==NULL || filename==NULL || config==NULL){
        printf("ERROR: dataset/filenmae/config=nuul");
        return 0;
    }

    pixel=config->input_w * config->input_h * config->input_ch;

    total_line=count_line_file(filename);
    if (total_line<=0){return 0;}

    capacity=total_line;
    if (max_images>0 && max_images<capacity){capacity=max_images;}

    dataset->images=(double *)safe_malloc(sizeof(double)*capacity*pixel);
    dataset->lable=(int *)safe_malloc(sizeof(int)*capacity);

    dataset->image_size=pixel;
    dataset->image_width=config->input_w;
    dataset->image_height=config->input_h;
    dataset->image_channels=config->input_ch;
    dataset->class_num=config->classes;

    file=fopen(filename, "r");
    if (file==NULL){printf("ERROR: file=null"); dataset_free(dataset); return 0; }

    image_index=0;
    while (fgets(line, sizeof(line), file)!=NULL){
        char *token;
        int label;
        int pixel_index;

        if (max_images>0 && image_index>=max_images){break;}
        
        token=strtok(line, ",");
        if (token==NULL){continue;}
        if (token[0]<'0' || token[0]>'9'){continue;}

        label=atoi(token);
        if (label<0 || label>=config->classes){
            printf("ERROR: incorrect label\n"); fclose(file); dataset_free(dataset); return 0;}

        pixel_index=0;

        //читаем пиксели после lable
        while ((token=strtok(NULL, ","))!=NULL){
            if (pixel_index>=pixel){
                printf("ERROR: too many pixels\n"); fclose(file); dataset_free(dataset); return 0;}
            
            int pixel_val=atoi(token); 
            dataset->images[image_index*pixel+pixel_index]=normal(pixel_val);
            pixel_index++;
        }
        if (pixel_index!=pixel){
            printf("ERROR: image has %d pixels, expected %d\n", pixel_index, pixel);
            fclose(file); dataset_free(dataset); return 0;
        }

        dataset->lable[image_index]=label;
        image_index++;
    }
    fclose(file);
    dataset->image_count=image_index;
    return 1;
}

//освобождаем память
void dataset_free(Dataset *dataset){
    if (dataset->images!=NULL){free(dataset->images); dataset->images=NULL;}
    if (dataset->lable!=NULL){free(dataset->lable); dataset->lable=NULL;}
    dataset->image_count=0;
    dataset->image_size=0;
    dataset->image_width=0;
    dataset->image_height=0;
    dataset->image_channels=0;
    dataset->class_num=0;
}

//печатаем информацию
void dataset_print(Dataset *dataset){
    printf("\n\nDataset info\n");
    printf("image_count: %d\n", dataset->image_count);
    printf("image_width: %d\n", dataset->image_width);
    printf("image_height: %d\n", dataset->image_height);
    printf("image_channels: %d\n", dataset->image_channels);
    printf("class_num: %d\n", dataset->class_num);
    printf("image_size: %d\n", dataset->image_size);
}
void dataset_print_labels(Dataset *dataset, int n){
    int limit=n;
    if (limit>dataset->image_count){limit=dataset->image_count;}
    printf ("labels %d:\n", limit);
    for (int i=0; i<limit; i++){ printf("%d ", dataset->lable[i]);}
}
void dataset_print_pixels(Dataset *dataset, int n){
    int limit=n;
    if (limit>dataset->image_size){ limit=dataset->image_size;}
    printf ("pixels %d:\n", limit);
    for (int i=0; i<limit; i++){ printf("%.3f", dataset->images[i]);}
}