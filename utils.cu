#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

FILE * load_dataset(const char * filename) {
    FILE * fp;
    fp = fopen(filename, "rb");
    return fp;
}

void close_dataset(FILE * fp) {
    fclose(fp);
}

float * get_image(FILE * fp, int image_idx) {
    int32_t rc[2];
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(int32_t), 2, fp);
    long size = rc[0] * rc[1];

    uint8_t pixels[size];
    fseek(fp, 16 + image_idx * size * sizeof(uint8_t), SEEK_SET);
    fread(pixels, sizeof(uint8_t), size, fp);

    float * image = (float *) malloc(size * sizeof(float));
    for(size_t i = 0; i < size; i++)
        image[i] = ((float)((int)pixels[i]))/255.0;
    
    return image;
}

int get_label(FILE * fp, int label_idx) {
    int32_t * rc[2];
    fseek(fp, 8, SEEK_SET);
    fread(rc, sizeof(int32_t), 2, fp);
    long size = ((int) rc[0]) * ((int) rc[1]);

    int label[1];
    fseek(fp, 8 + label_idx * sizeof(uint8_t), SEEK_SET);
    fread(label, sizeof(uint8_t), 1, fp);

    return label[0];
}