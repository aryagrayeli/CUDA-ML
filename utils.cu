#include <stdio.h>

FILE * load_dataset(const char * filename) {
    FILE * fp;
    fp = fopen(filename, "rb");
    return fp;
}

void close_dataset(FILE * fp) {
    fclose(fp);
}

float * get_image(FILE * fp, int image_idx) {

}

int get_label(FILE * fp, int label_idx) {
    
}