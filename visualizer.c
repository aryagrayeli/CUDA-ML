
#include "utils.h"

int main(int argc, char **argv) {

    FILE * dataset = load_dataset("train_data/train-images.idx3-ubyte");
    FILE * labels = load_dataset("train_data/train-labels.idx1-ubyte");

    int idx = 10;
    double * image = get_image(dataset, idx);
    uint8_t label = get_label(labels, idx);

    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            if(image[28*i+j] > 0.01)
                printf(". ");
            else
                printf("  ");
        }
        printf("\n");
    }
    printf("\n");

    printf("%d\n\n", label);

}