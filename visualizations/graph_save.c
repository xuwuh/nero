#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph_save.h"

#define MAX_EPOCHS 512
#define MAX_CLASSES 10

typedef struct
{
    int count;
    int epoch[MAX_EPOCHS];
    double train_loss[MAX_EPOCHS];
    double train_accuracy[MAX_EPOCHS];
    double test_loss[MAX_EPOCHS];
    double test_accuracy[MAX_EPOCHS];
    double epoch_time[MAX_EPOCHS];
} GraphHistory;

//чтение истории обучения
static int graph_read_history(char *filename, GraphHistory *history){
    FILE *file=fopen(filename, "r");
    char line[512];
    if (file==NULL){return 0;}

    fgets(line, sizeof(line), file);
    history->count=0;
    while (fgets(line, sizeof(line), file)!=NULL && history->count<MAX_EPOCHS){
        int i=history->count;
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf", &history->epoch[i], &history->train_loss[i], &history->train_accuracy[i], &history->test_loss[i], &history->test_accuracy[i], &history->epoch_time[i])==6){
            history->count++;
        }
    }

    fclose(file);
    return history->count>0;
}

//чтение confusion matrix
static int graph_read_confusion(char *filename, int matrix[MAX_CLASSES][MAX_CLASSES], int *class_count){
    FILE *file=fopen(filename, "r");
    char line[512];
    if (file==NULL){return 0;}

    if (fgets(line, sizeof(line), file)==NULL){fclose(file); return 0;}
    *class_count=0;

    while (fgets(line, sizeof(line), file)!=NULL && *class_count<MAX_CLASSES){
        char *token=strtok(line, ",");
        int row=*class_count;
        int col=0;
        token=strtok(NULL, ",");
        while (token!=NULL && col<MAX_CLASSES){
            matrix[row][col]=atoi(token);
            token=strtok(NULL, ",");
            col++;
        }
        (*class_count)++;
    }

    fclose(file);
    return *class_count>0;
}

static double graph_max4(double *a, double *b, double *c, double *d, int count){
    double max_value=a[0];
    for (int i=0; i<count; i++){
        if (a[i]>max_value){max_value=a[i];}
        if (b[i]>max_value){max_value=b[i];}
        if (c[i]>max_value){max_value=c[i];}
        if (d[i]>max_value){max_value=d[i];}
    }
    return max_value;
}

static double graph_max2(double *a, double *b, int count){
    double max_value=a[0];
    for (int i=0; i<count; i++){
        if (a[i]>max_value){max_value=a[i];}
        if (b[i]>max_value){max_value=b[i];}
    }
    return max_value;
}

static double graph_x(int index, int count){
    return 70.0+(double)index*780.0/(double)(count>1 ? count-1 : 1);
}

static double graph_y(double value, double min_value, double max_value){
    return 390.0-(value-min_value)*320.0/(max_value-min_value);
}

static void graph_polyline(FILE *file, int count, double *values, double min_value, double max_value, char *color){
    fprintf(file, "<polyline points=\"");
    for (int i=0; i<count; i++){
        fprintf(file, "%.2f,%.2f ", graph_x(i, count), graph_y(values[i], min_value, max_value));
    }
    fprintf(file, "\" fill=\"none\" stroke=\"%s\" stroke-width=\"3\"/>\n", color);
}

//линейный график
static int graph_save_line(char *filename, char *title, GraphHistory *sgd, GraphHistory *momentum, int type){
    FILE *file=fopen(filename, "w");
    int count=sgd->count<momentum->count ? sgd->count : momentum->count;
    double max_value;
    double min_value=0.0;
    if (file==NULL || count<=0){return 0;}

    if (type==0){max_value=graph_max4(sgd->train_loss, sgd->test_loss, momentum->train_loss, momentum->test_loss, count);}
    else if (type==1){max_value=1.0;}
    else{max_value=graph_max2(sgd->epoch_time, momentum->epoch_time, count);}
    if (max_value<=min_value){max_value=min_value+1.0;}

    fprintf(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"460\" viewBox=\"0 0 900 460\">\n");
    fprintf(file, "<rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n");
    fprintf(file, "<text x=\"450\" y=\"32\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"22\" font-weight=\"bold\">%s</text>\n", title);
    fprintf(file, "<line x1=\"70\" y1=\"390\" x2=\"850\" y2=\"390\" stroke=\"#789\"/>\n");
    fprintf(file, "<line x1=\"70\" y1=\"70\" x2=\"70\" y2=\"390\" stroke=\"#789\"/>\n");

    if (type==0){
        graph_polyline(file, count, sgd->train_loss, min_value, max_value, "#2779bd");
        graph_polyline(file, count, sgd->test_loss, min_value, max_value, "#7cc0e8");
        graph_polyline(file, count, momentum->train_loss, min_value, max_value, "#c75c5c");
        graph_polyline(file, count, momentum->test_loss, min_value, max_value, "#f08a8a");
    }
    else if (type==1){
        graph_polyline(file, count, sgd->train_accuracy, min_value, max_value, "#2779bd");
        graph_polyline(file, count, sgd->test_accuracy, min_value, max_value, "#7cc0e8");
        graph_polyline(file, count, momentum->train_accuracy, min_value, max_value, "#c75c5c");
        graph_polyline(file, count, momentum->test_accuracy, min_value, max_value, "#f08a8a");
    }
    else{
        graph_polyline(file, count, sgd->epoch_time, min_value, max_value, "#2779bd");
        graph_polyline(file, count, momentum->epoch_time, min_value, max_value, "#c75c5c");
    }

    fprintf(file, "<text x=\"80\" y=\"430\" font-family=\"Arial\" font-size=\"13\" fill=\"#2779bd\">SGD</text>\n");
    fprintf(file, "<text x=\"150\" y=\"430\" font-family=\"Arial\" font-size=\"13\" fill=\"#c75c5c\">Momentum</text>\n");
    fprintf(file, "</svg>\n");
    fclose(file);
    return 1;
}

//столбчатый график confusion matrix
static int graph_save_confusion(char *input_file, char *output_file, char *title){
    FILE *file=fopen(output_file, "w");
    int matrix[MAX_CLASSES][MAX_CLASSES]={0};
    int class_count=0;
    int max_value=1;

    if (file==NULL){return 0;}
    if (!graph_read_confusion(input_file, matrix, &class_count)){fclose(file); return 0;}

    for (int i=0; i<class_count; i++){
        int row_sum=0;
        for (int j=0; j<class_count; j++){row_sum+=matrix[i][j];}
        if (row_sum>max_value){max_value=row_sum;}
    }

    fprintf(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"460\" viewBox=\"0 0 900 460\">\n");
    fprintf(file, "<rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n");
    fprintf(file, "<text x=\"450\" y=\"32\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"22\" font-weight=\"bold\">%s</text>\n", title);
    fprintf(file, "<line x1=\"70\" y1=\"390\" x2=\"850\" y2=\"390\" stroke=\"#789\"/>\n");

    for (int i=0; i<class_count; i++){
        int row_sum=0;
        int correct=matrix[i][i];
        int wrong;
        double x=80.0+i*76.0;
        double correct_h;
        double wrong_h;

        for (int j=0; j<class_count; j++){row_sum+=matrix[i][j];}
        wrong=row_sum-correct;
        correct_h=(double)correct*300.0/(double)max_value;
        wrong_h=(double)wrong*300.0/(double)max_value;

        fprintf(file, "<rect x=\"%.2f\" y=\"%.2f\" width=\"46\" height=\"%.2f\" fill=\"#4f9fd9\"/>\n", x, 390.0-correct_h, correct_h);
        fprintf(file, "<rect x=\"%.2f\" y=\"%.2f\" width=\"46\" height=\"%.2f\" fill=\"#f08a8a\"/>\n", x, 390.0-correct_h-wrong_h, wrong_h);
        fprintf(file, "<text x=\"%.2f\" y=\"420\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"13\">%c</text>\n", x+23.0, 'A'+i);
    }

    fprintf(file, "<text x=\"70\" y=\"445\" font-family=\"Arial\" font-size=\"13\" fill=\"#4f9fd9\">correct</text>\n");
    fprintf(file, "<text x=\"150\" y=\"445\" font-family=\"Arial\" font-size=\"13\" fill=\"#f08a8a\">wrong</text>\n");
    fprintf(file, "</svg>\n");
    fclose(file);
    return 1;
}

//сохранение всех графиков
int graph_save_all(void){
    GraphHistory sgd;
    GraphHistory momentum;
    system("if not exist visualizations\\out mkdir visualizations\\out");

    if (!graph_read_history("results/history_sgd.csv", &sgd)){return 0;}
    if (!graph_read_history("results/history_momentum.csv", &momentum)){return 0;}

    graph_save_line("visualizations/out/loss_by_epoch.svg", "Loss by epoch", &sgd, &momentum, 0);
    graph_save_line("visualizations/out/accuracy_by_epoch.svg", "Accuracy by epoch", &sgd, &momentum, 1);
    graph_save_line("visualizations/out/epoch_time.svg", "Epoch time", &sgd, &momentum, 2);
    graph_save_confusion("results/confusion_sgd.csv", "visualizations/out/confusion_sgd_bars.svg", "SGD confusion matrix by class");
    graph_save_confusion("results/confusion_momentum.csv", "visualizations/out/confusion_momentum_bars.svg", "Momentum confusion matrix by class");
    return 1;
}
