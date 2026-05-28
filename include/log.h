#ifndef LOG_H
#define LOG_H

typedef struct
{
    double final_train_loss;
    double final_train_accuracy;
    double final_test_loss;
    double final_test_accuracy;
    double total_time;
    double average_epoch_time;
} Log;

//создание history csv
int log_init_history(char *filename);

//запись эпохи
int log_log_epoch(char *filename, int epoch, double train_loss, double train_accuracy, double test_loss, double test_accuracy, double epoch_time);

//создание comparison csv
int log_init_comparison(char *filename);

//запись сравнения
int log_log_comparison(char *filename, char *mode, Log *summary);

//сохранение confusion matrix
int log_save_confusion_matrix(char *filename, int *matrix, int class_num);

#endif
