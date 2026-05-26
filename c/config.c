#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "../include/config.h"

//убираем пробелы и прочие лишние символы в строке
void del(char *str){
    char *start=str, *end;
    while (isspace((char)*start)){start++;}
    if (*start=='\0'){str[0]='\0'; return;}
    end=start+strlen(start)-1;
    while (end>start&&isspace((char)*end)){end--;}
    *(end+1)='\0';
    if (start!=str){memmove(str, start, strlen(start)+1);}
}

//переводим строку в вид ключ+значение
int key_value(Config *config, char *key, char *value){
    if (strcmp(key, "input_w")==0){config->input_w=atoi(value);}
    else if (strcmp(key, "input_h")==0){config->input_h=atoi(value);}
    else if (strcmp(key, "input_ch")==0){config->input_ch=atoi(value);}
    else if (strcmp(key, "classes")==0){config->classes=atoi(value);}
    else if (strcmp(key, "epochs")==0){config->epochs=atoi(value);}
    else if (strcmp(key, "batch_size")==0){config->batch_size=atoi(value);}
    else if (strcmp(key, "momentum")==0){config->momentum=atof(value);}
    else if (strcmp(key, "learning_rate")==0){config->learning_rate=atof(value);}
    else if (strcmp(key, "regularizator")==0){config->regularizator=atof(value);}

    else if (strcmp(key, "train_data_path")==0){
        strncpy(config->train_data_path, value, 256-1);
        config->train_data_path[256-1]='\0';}
    else if (strcmp(key, "test_data_path")==0){
        strncpy(config->test_data_path, value, 256-1);
        config->test_data_path[256-1]='\0';}

    else {printf("неизвесное значение в конфиге '%s' ", key);}
    return 1;
}

//читаем файл
int load(char *filename, Config *config){
    if (filename==NULL || config==NULL){ return 0;}
    memset(config, 0, sizeof(*config));

    FILE *file;
    char line[512];
    file=fopen(filename, "r");
    if (file==NULL){ printf("ERROR: cannon open config file\n"); return 0;}

    while (fgets(line, sizeof(line), file)!=NULL){
        char *pos, key[256], value[256];
        del(line);
        if (line[0]=='\0'){continue;}

        pos=strchr(line, ':');
        if (pos==NULL){continue;}
        *pos='\0';

        snprintf(key, sizeof(key), "%s", line);
        snprintf(value, sizeof(value), "%s", pos+1);
        del(key);
        del(value);
        key_value(config, key, value);
    }
    fclose(file);
    return validate(config);
}

//проверяем коректность значений
int validate(Config *config){
    if (config==NULL){return 0;}
    if (config->input_w<=0 || config->input_h<=0){printf("Warning: input_w или input_h\n"); return 0;}
    else if (config->input_ch<=0){printf("Warning: input_ch\n"); return 0;}
    else if (config->classes<=0){printf("Warning: classes\n"); return 0;}
    else if (config->epochs<=0){printf("Warning: epochs\n"); return 0;}
    else if (config->batch_size<=0){printf("Warning: batch_size\n"); return 0;}
    else if (config->momentum<=0.0 || config->momentum>=1.0){printf("Warning: momentum\n"); return 0;}
    else if (config->learning_rate<=0.0){printf("Warning: learning_rate\n"); return 0;}
    else if (config->regularizator<0.0){printf("Warning: regularizator\n"); return 0;}
    return 1;
}

//печатаем конфиг для проверки коректности чтения данных
void conf_print(Config *config){
    printf("\n\nCONFIG\n");
    printf("input_w: %d\n", config->input_w);
    printf("input_h: %d\n", config->input_h);
    printf("input_ch: %d\n", config->input_ch);
    printf("classes: %d\n", config->classes);
    printf("epochs: %d\n", config->epochs);
    printf("batch_size: %d\n", config->batch_size);
    printf("momentum: %.2f\n", config->momentum);
    printf("learning_rate: %.2f\n", config->learning_rate);
    printf("regularizator: %.4f\n", config->regularizator);

    printf("train_data_path: %s\n", config->train_data_path);
    printf("test_data_path: %s\n", config->test_data_path);
}
