#ifndef TENSOR_H
#define TENSOR_H

typedef struct
{
    int n; //кол-во обьктов в батче
    int c; //кол-во каналов
    int h; //высота
    int w; //ширина
    int size; //кол-во элементов всего
    double *data; //все значения
} Tensor;

//создание тензора
Tensor tensor_create(int n, int c, int h, int w);
//освободить память
void tensor_free(Tensor *tensor);
//заполнить нулями
void tensor_zero(Tensor *tensor);
//заполнить одним значением
void tensor_value(Tensor *tensor, double value);
//получить элемент
double tensor_get(const Tensor *tensor, int n, int c, int h, int w);
//установить элемент 
void tensor_set(Tensor *tensor, int n, int c, int h, int w, double value);
//копирование тензора
int tensor_copy(const Tensor *src, Tensor *dst);
//перевод батча из матриц в тензор 
int tensor_from_images(Tensor *tensor, const double *flat_images, int image_count, int image_size);

//печать и тесты
void tensor_print(const Tensor *tensor, const char *name);
void tensor_tests();

#endif 