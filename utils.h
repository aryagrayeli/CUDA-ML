#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// This file are helper and I/O methods for dealing with the MNIST dataset format

char* concat(char * s1, const char * s2) {
    char * result = (char *) malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

FILE * load_dataset(char * filename) {
    FILE * fp;
    fp = fopen(filename, "rb");
    free(filename);
    return fp;
}

void close_dataset(FILE * fp) {
    fclose(fp);
}

uint32_t get_dataset_size(FILE * fp) {
    uint8_t * rc = (uint8_t *) malloc(sizeof(uint8_t) * 4);
    fseek(fp, 4, SEEK_SET);
    fread(rc, sizeof(uint8_t), 4, fp);
    uint32_t size = ((uint32_t) rc[0] << 24) | ((uint32_t) rc[1] << 16) | ((uint32_t) rc[2] << 8) | ((uint32_t) rc[3]);
    free(rc);
    return size;
}

uint32_t get_image_width(FILE * fp) {
    uint8_t * rc = (uint8_t *) malloc(sizeof(uint8_t) * 8);
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(uint8_t), 8, fp);
    uint32_t dim2 = ((uint32_t) rc[4] << 24) | ((uint32_t) rc[5] << 16) | ((uint32_t) rc[6] << 8) | ((uint32_t) rc[7]);
    free(rc);
    return dim2;
}

uint32_t get_image_height(FILE * fp) {
    uint8_t * rc = (uint8_t *) malloc(sizeof(uint8_t) * 8);
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(uint8_t), 8, fp);
    uint32_t dim1 = ((uint32_t) rc[0] << 24) | ((uint32_t) rc[1] << 16) | ((uint32_t) rc[2] << 8) | ((uint32_t) rc[3]);
    free(rc);
    return dim1;
}

double * get_image(FILE * fp, int image_idx) {
    uint8_t * rc = (uint8_t *) malloc(sizeof(uint8_t) * 8);
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(uint8_t), 8, fp);
    uint32_t dim1 = ((uint32_t) rc[0] << 24) | ((uint32_t) rc[1] << 16) | ((uint32_t) rc[2] << 8) | ((uint32_t) rc[3]);
    uint32_t dim2 = ((uint32_t) rc[4] << 24) | ((uint32_t) rc[5] << 16) | ((uint32_t) rc[6] << 8) | ((uint32_t) rc[7]);
    uint32_t size = dim1 * dim2;
    free(rc);

    uint8_t pixels[size];
    fseek(fp, 16 + image_idx * size * sizeof(uint8_t), SEEK_SET);
    fread(pixels, sizeof(uint8_t), size, fp);

    double * image = (double *) malloc(size * sizeof(double));
    for(size_t i = 0; i < size; i++)
        image[i] = ((double)((int)pixels[i]))/255.0;
    
    return image;
}

uint8_t get_label(FILE * fp, int label_idx) {
    uint8_t label[1];
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

double * load_labels(FILE * labels, int * dataloader, int idx, int batch_size, uint64_t num_classes) {
    double * true_y = (double *) malloc(sizeof(double)*num_classes*batch_size);
    for(int i = 0; i < batch_size; i++) {
        uint8_t l = get_label(labels, dataloader[idx+i]);
        for(int j = 0; j < num_classes; j++) {
            if(j == l)
                true_y[num_classes*i + j] = 1.0;
            else
                true_y[num_classes*i + j] = 0.0;
        }
    }
    return true_y;
}

int arg_max(double * y, int idx, uint64_t num_classes) {
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