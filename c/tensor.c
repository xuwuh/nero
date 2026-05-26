#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"
#include "../include/utils.h"

//переводим издексы в плоский массив 
int tensor_index(const Tensor *tensor, int n, int c, int h, int w){
    return ((n*tensor->c+c)*tensor->h+h)*tensor->w+w;
}

//создание тензора
Tensor tensor_create(int n, int c, int h, int w){
    Tensor tensor;
    tensor.n=n;
    tensor.c=c;
    tensor.h=h;
    tensor.w=w;
    tensor.size=0;
    tensor.data=NULL;

    tensor.size=n*c*h*w;
    tensor.data=(double *)safe_calloc(tensor.size, sizeof(double));
    return tensor;
}

//освободить память
void tensor_free(Tensor *tensor){
    if (tensor==NULL){return;}
    if (tensor->data!=NULL){
        free(tensor->data);
        tensor->data=NULL;
    }
    tensor->n=0;
    tensor->c=0;
    tensor->h=0;
    tensor->w=0;
    tensor->size=0;
}

//заполнить нулями
void tensor_zero(Tensor *tensor){
    for (int i=0; i<tensor->size; i++){tensor->data[i]=0.0;}
}

//заполнить одним значением
void tensor_value(Tensor *tensor, double value){
    for (int i=0; i<tensor->size; i++){tensor->data[i]=value;}
}

//получить элемент
double tensor_get(const Tensor *tensor, int n, int c, int h, int w){
    int i=tensor_index(tensor, n, c, h, w);
    return tensor->data[i];
}

//установить элемент 
void tensor_set(Tensor *tensor, int n, int c, int h, int w, double value){
    int i=tensor_index(tensor, n, c, h, w);
    tensor->data[i]=value;
}

//копирование тензора
int tensor_copy(const Tensor *src, Tensor *dst){
    for (int i = 0; i < src->size; i++){dst->data[i] = src->data[i];}
    return 1;
}

//перевод батча из матриц в тензор 
int tensor_from_images(Tensor *tensor, const double *flat_images, int image_count, int image_size){
    for (int n=0; n<image_count; n++){
        for (int i=0; i<image_size; i++){
            tensor->data[n*image_size+i]=flat_images[n*image_size+i];
        }
    }
    return 1;
}

//
void tensor_print(const Tensor *tensor, const char *name){
    if (tensor==NULL){printf("%s: NULL tensor\n", name); return;}
    printf("%s: [%d x %d x %d x %d], size=%d\n", name, tensor->n, tensor->c, tensor->h, tensor->w, tensor->size);
}

void tensor_tests(){
    Tensor tensor;
    Tensor copy;

    printf("\nTENSOR TEST\n");

    tensor=tensor_create(2, 1, 2, 3);
    tensor_print(&tensor, "test tensor");

    tensor_set(&tensor, 0, 0, 0, 0, 1.0);
    tensor_set(&tensor, 0, 0, 0, 1, 2.0);
    tensor_set(&tensor, 1, 0, 1, 2, 9.0);
    printf("tensor[0][0][0][0]=%.2f\n", tensor_get(&tensor, 0, 0, 0, 0));
    printf("tensor[0][0][0][1]=%.2f\n", tensor_get(&tensor, 0, 0, 0, 1));
    printf("tensor[1][0][1][2]=%.2f\n", tensor_get(&tensor, 1, 0, 1, 2));

    copy=tensor_create(2, 1, 2, 3);
    tensor_copy(&tensor, &copy);
    printf("copy[1][0][1][2]=%.2f\n", tensor_get(&copy, 1, 0, 1, 2));
    
    tensor_free(&tensor);
    tensor_free(&copy);
}