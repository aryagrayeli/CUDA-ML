#pragma once

#include <stdio.h>
#include <stdint.h>

#include "utils.h"

typedef struct DatasetInfo {
  char * train_files;
  char * test_files;
  char * checkpoint_path;
  uint64_t epochs;
  uint64_t batch_size;
  char * loss_func;
} DatasetInfo;

typedef struct ArchInfo {
  uint64_t layers;
  uint64_t * layers_size;
  char ** activation_function;
} ArchInfo;

typedef struct Model {
    float ** weights; // list of 2D weight matrix for each layer but flattened so 1D for CUDA
    float ** biases; // list of 1D bias vectors for each layer
    ArchInfo * arch_info; // architecture info
} Model;

Model * initialize_model(ArchInfo * arch_info) {
    Model * model = (Model *) malloc(sizeof(Model));

    model->weights = (float **) malloc(sizeof(float*) * (arch_info->layers-1));
    for(int i = 0; i < arch_info->layers-1; i++) {
        uint64_t n = arch_info->layers_size[i];
        uint64_t m = arch_info->layers_size[i+1];
        model->weights[i] = (float *) malloc(sizeof(float) * n * m);

        // gpu initialize model->weights[i] in parallel
        // different initialization depending on the activation function
        char * act_func = arch_info->activation_function[i];

        // if(strcmp(act_func, "ReLU") == 0)
        //     he_initialization = G(0.0, sqrt(2/n))
        // else
        //     xavier_initialization = U(-sqrt(6/(n+m)), sqrt(6/(n+m)))
    }

    model->biases = (float **) malloc(sizeof(float*) * (arch_info->layers-1));
    for(int i = 0; i < arch_info->layers-1; i++) {
        uint64_t m = arch_info->layers_size[i+1];
        model->biases[i] = (float *) malloc(sizeof(float) * m);

        // gpu initialize model->biases[i] in parallel
        // initialize all to 0
    }

    model->arch_info = arch_info;
    return model;
}

void save_model(Model * model, char * checkpoint_path) {
    // TODO
}

Model * load_model(char * checkpoint_path, ArchInfo * arch_info) {
    // TODO
    return initialize_model(arch_info);
}

int * forward(Model * model, FILE * dataset, int * dataloader, int idx, int batch_size) {
    // TODO
    int * pred_y = (int *) malloc(sizeof(int)*batch_size);
    for(int i = 0; i < batch_size; i++) {
        pred_y[i] = i%10;
    }
    return pred_y;
}

float backward(int * pred_y, int * true_y, Model * model, char * loss_func, int batch_size) {
    // TODO
    return 0.1; // loss value
}


void train(DatasetInfo * dataset_info, ArchInfo * arch_info) {
    printf("Beginning Training\n");

    FILE * dataset = load_dataset(concat(dataset_info->train_files,"/train-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info->train_files,"/train-labels.idx1-ubyte"));
    int32_t num_images = get_dataset_size(dataset);
    num_images = 16;

    int * dataloader = (int *) malloc(sizeof(int) * num_images);
    for(int i = 0; i < num_images; i++) {
        dataloader[i] = i;
    }

    Model * model = initialize_model(arch_info);

    printf("Dataset and Model Initalized\n");

    for(int e = 0; e < dataset_info->epochs; e++) {
        shuffle(dataloader, num_images);

        float running_loss = 0.0;
        for(int idx = 0; idx < num_images; idx+=dataset_info->batch_size) {
            int * pred_y = forward(model, dataset, dataloader, idx, dataset_info->batch_size);
            int * true_y = load_labels(labels, dataloader, idx, dataset_info->batch_size);
            float loss = backward(pred_y, true_y, model, dataset_info->loss_func, dataset_info->batch_size);
            running_loss+=loss;
        }
        printf("After Epoch %d, Train Loss: %f\n", e, running_loss);
    }

    save_model(model, dataset_info->checkpoint_path);
    close_dataset(dataset);
    close_dataset(labels);

    printf("Saved Model\n\n");
}

void test(DatasetInfo * dataset_info, ArchInfo * arch_info) {
    printf("Beginning Testing\n");

    FILE * dataset = load_dataset(concat(dataset_info->test_files,"/test-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info->test_files,"/test-labels.idx1-ubyte"));
    int32_t num_images = get_dataset_size(dataset);
    num_images = 16;
    
    Model * model = load_model(dataset_info->checkpoint_path, arch_info);

    int * dataloader = (int *) malloc(sizeof(int) * num_images);
    for(int i = 0; i < num_images; i++) {
        dataloader[i] = i;
    }

    printf("Dataset and Model Initalized\n");

    float accuracy = 0.0;
    for(int idx = 0; idx < num_images; idx+=dataset_info->batch_size) {
        int * pred_y = forward(model, dataset, dataloader, idx, dataset_info->batch_size);
        int * true_y = load_labels(labels, dataloader, idx, dataset_info->batch_size);
        
        for(int i = 0; i < dataset_info->batch_size; i++) {
            if(pred_y[i] == true_y[i])
                accuracy++;
        }
    }

    accuracy/=num_images;
    printf("Test Accuracy of %f for %d images\n\n", accuracy, num_images);

    close_dataset(dataset);
    close_dataset(labels);
}

void predict(DatasetInfo * dataset_info, ArchInfo * arch_info, int image_idx) {
    printf("Beginning Prediction on Image #%d\n", image_idx);
    
    FILE * dataset = load_dataset(concat(dataset_info->test_files,"/test-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info->test_files,"/test-labels.idx1-ubyte"));

    Model * model = load_model(dataset_info->checkpoint_path, arch_info);

    int dataloader[1];
    dataloader[0] = image_idx;

    printf("Image and Model Initalized\n");

    int * pred_y = forward(model, dataset, dataloader, 0, 1);
    int * true_y = load_labels(labels, dataloader, 0, 1);
    printf("Predicted: %d, Actual: %d\n\n", pred_y[0], true_y[0]);

    close_dataset(dataset);
    close_dataset(labels);
}