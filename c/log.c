#include <stdio.h>

#include "../include/log.h"

//создание history csv
int log_init_history(char *filename){
    FILE *file;
    file=fopen(filename, "w");
    fprintf(file, "epoch, train_loss, train_accuracy, test_loss, test_accuracy, epoch_time\n");
    fclose(file);
    return 1;
}

//запись эпохи
int log_log_epoch(char *filename, int epoch, double train_loss, double train_accuracy, double test_loss, double test_accuracy, double epoch_time){
    FILE *file;
    file=fopen(filename, "a");
    fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", epoch, train_loss, train_accuracy, test_loss, test_accuracy, epoch_time);
    fclose(file);
    return 1;
}

//создание comparison csv
int log_init_comparison(char *filename){
    FILE *file;
    file=fopen(filename, "w");
    fprintf(file, "mode, final_train_loss, final_train_accuracy, final_test_loss, final_test_accuracy, average_epoch_time, total_time\n");
    fclose(file);
    return 1;
}

//запись сравнения
int log_log_comparison(char *filename, char *mode, Log *summary){
    FILE *file;
    file=fopen(filename, "a");
    fprintf(file, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", mode, summary->final_train_loss, summary->final_train_accuracy, summary->final_test_loss, summary->final_test_accuracy, summary->average_epoch_time, summary->total_time);
    fclose(file);
    return 1;
}

//сохранение confusion matrix
int log_save_confusion_matrix(char *filename, int *matrix, int class_num){
    FILE *file;
    file=fopen(filename, "w");

    fprintf(file, "true_pred");
    for (int j=0; j<class_num; j++){
        fprintf(file, ",%d", j);
    }
    fprintf(file, "\n");

    for (int i=0; i<class_num; i++){
        fprintf(file, "%d", i);
        for (int j=0; j<class_num; j++){
            fprintf(file, ",%d", matrix[i*class_num+j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 1;
}
