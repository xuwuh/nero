#ifndef UTILS_H
#define UTILS_H
#include <stddef.h>

//выделение памяти
void *safe_malloc(size_t size);
//выделение памяти с обнулением
void *safe_calloc (size_t count, size_t size);
//ограничение дабл диапазоном
double clamp_double (double value, double min_val, double max_val);
//нандомное число
double random_double (double min_ranval, double max_ranval);
//рандомный вес
double random_weight (double scale);
// индекс максимального числа массива
int max_arg(double *values, int count);

//тесты 
void utils_test();

#endif 