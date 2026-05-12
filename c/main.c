#include <stdio.h>

#include "../include/config.h"

int main(){
    Config config;

    load("configs/config.txt", &config);
    conf_print(&config);

    return 0;
}