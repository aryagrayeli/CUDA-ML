#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define num_classes (10)


char* concat(char * s1, char * s2) {
    char * result = (char *) malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

FILE * load_dataset(char * filename) {
    FILE * fp;
    fp = fopen(filename, "rb");
    return fp;
}

void close_dataset(FILE * fp) {
    fclose(fp);
}

int32_t get_dataset_size(FILE * fp) {
    int32_t * size;
    fseek(fp, 4, SEEK_SET);
    fread(size, sizeof(int32_t), 1, fp);
    return *size;
}

double * get_image(FILE * fp, int image_idx) {
    int32_t rc[2];
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(int32_t), 2, fp);
    long size = rc[0] * rc[1];

    uint8_t pixels[size];
    fseek(fp, 16 + image_idx * size * sizeof(uint8_t), SEEK_SET);
    fread(pixels, sizeof(uint8_t), size, fp);

    double * image = (double *) malloc(size * sizeof(double));
    for(size_t i = 0; i < size; i++)
        image[i] = ((double)((int)pixels[i]))/255.0;
    
    return image;
}

int get_label(FILE * fp, int label_idx) {
    int32_t rc[2];
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(int32_t), 2, fp);
    long size = rc[0] * rc[1];

    int label[1];
    fseek(fp, 8 + label_idx * sizeof(uint8_t), SEEK_SET);
    fread(label, sizeof(uint8_t), 1, fp);

    return label[0];
}

void shuffle(int * array, int32_t n) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

double * load_labels(FILE * labels, int * dataloader, int idx, int batch_size) {
    double * true_y = (double *) malloc(sizeof(double)*num_classes*batch_size);
    for(int i = 0; i < batch_size; i++) {
        int l = get_label(labels, dataloader[idx+i]);
        for(int j = 0; j < num_classes; j++) {
            if(j == l)
                true_y[num_classes*i + j] = 1.0;
            else
                true_y[num_classes*i + j] = 0.0;
        }
    }
    return true_y;
}

int arg_max(double * y, int idx) {
    double max_val = 0.0;
    int max_idx = 0;

    for(int i = 0; i < num_classes; i++) {
        double v = y[idx*num_classes + i];
        if(v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return max_idx;
}