#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/utils.h"

//выделение памяти
void *safe_malloc(size_t size){
    void *ptr;
    ptr=malloc(size);
    if (ptr==NULL){printf("ERROR: MALLOC"); exit(1);}
    return ptr;
}

//выделение памяти с обнулением
void *safe_calloc (size_t count, size_t size){
    void *ptr;
    ptr=calloc(count, size);
    if (ptr==NULL){printf("ERROR: CALLOC"); exit(1);}
    return ptr;
}

//ограничение дабл диапазоном
double clamp_double (double value, double min_val, double max_val){
    if (value<min_val){return min_val;}
    if (value>max_val){return max_val;}
    return value;
}

//нандомное число
double random_double (double min_ranval, double max_ranval){
    double r=(double)rand()/(double)RAND_MAX;
    return min_ranval+r*(max_ranval-min_ranval);
}

//рандомный вес
double random_weight (double scale){
    return random_double(-scale, scale);
}

// индекс максимального числа массива
int max_arg(double *values, int count){
    int best_i=0; 
    double best_val=values[0];
    for (int i=0; i<count; i++){
        if (values[i]>best_val){
            best_i=i;
            best_val=values[i];
        }
    }
    return best_i;
}

//время
double check_time(){
    return (double)clock()/CLOCKS_PER_SEC;
}

