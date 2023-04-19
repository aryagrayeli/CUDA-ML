#pragma once

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "utils.h"
#include "matmul.cu"

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
    double ** weights; // list of 2D weight matrix for each layer but flattened so 1D for CUDA
    double ** biases; // list of 1D bias vectors for each layer
    ArchInfo * arch_info; // architecture info
} Model;

Model * initialize_model(ArchInfo * arch_info) {
    Model * model = (Model *) malloc(sizeof(Model));

    model->weights = (double **) malloc(sizeof(double*) * (arch_info->layers-1));
    for(int i = 0; i < arch_info->layers-1; i++) {
        double n = (double) arch_info->layers_size[i];
        double m = (double) arch_info->layers_size[i+1];
        cudaMalloc(&(model->weights[i]), sizeof(double) * n * m);

        // gpu initialize model->weights[i] in parallel
        // different initialization depending on the activation function
        char * act_func = arch_info->activation_function[i];

        dim3 gridSz(1, 1, 1);
        dim3 blockSz(n*m, 1, 1);

        if(strcmp(act_func, "ReLU") == 0)
            he_init<<<gridSz, blockSz>>>(model->weights[i], sqrt(2.0/n), (int) n*m); // values sampled from G(0.0, sqrt(2/n))
        else
            xavier_init<<<gridSz, blockSz>>>(model->weights[i], sqrt(6.0/(n+m)), (int) n*m); // values sampled from U(-sqrt(6/(n+m)), sqrt(6/(n+m)))
    }

    model->biases = (double **) malloc(sizeof(double*) * (arch_info->layers-1));
    for(int i = 0; i < arch_info->layers-1; i++) {
        int m = (int) arch_info->layers_size[i+1];
        cudaMalloc(&(model->biases[i]), sizeof(double) * m);

        dim3 gridSz(1, 1, 1);
        dim3 blockSz(m, 1, 1);

        // gpu initialize model->biases[i] in parallel
        zero_init<<<gridSz, blockSz>>>(model->biases[i], m);
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

double * forward(Model * model, FILE * dataset, int * dataloader, int idx, int batch_size) {
    double ** weights = model->weights;
    double ** biases = model->biases;
    ArchInfo * arch_info = model->arch_info;

    // load and flatten batches
    uint64_t size = arch_info->layers_size[0];
    double * input;
    cudaMalloc(&input, sizeof(double) * size * batch_size);
    for(int i = 0; i < batch_size; i++) {
        double * pixels = get_image(dataset, dataloader[idx+i]);
        cudaMemcpy(input + (i * size), pixels, sizeof(double) * size, cudaMemcpyHostToDevice);
    }

    for(int l = 0; l < arch_info->layers-1; l++) {
        int prev_size = (int) arch_info->layers_size[l];
        int next_size = (int) arch_info->layers_size[l+1];

        double * output;
        cudaMalloc(&output, sizeof(double) * next_size * batch_size);

        dim3 gridSz(1,1,1);
        dim3 blockSz(batch_size,1,1);

        batch_matrix_mul<<<gridSz,blockSz>>>(weights[l], input, output, next_size, prev_size, batch_size);
        batch_vector_add<<<gridSz,blockSz>>>(output, biases[l], output, next_size, batch_size);
        
        char * act = arch_info->activation_function[l];
        if(strcmp(act, "ReLU") == 0)
            batch_vector_relu<<<gridSz,blockSz>>>(output, output, next_size, batch_size);
        if(strcmp(act, "sigmoid") == 0)
            batch_vector_sigmoid<<<gridSz,blockSz>>>(output, output, next_size, batch_size);
        if(strcmp(act, "tanh") == 0)
            batch_vector_tanh<<<gridSz,blockSz>>>(output, output, next_size, batch_size);
        if(strcmp(act, "softmax") == 0)
            batch_vector_softmax<<<gridSz,blockSz>>>(output, output, next_size, batch_size);

        cudaFree(input);
        input = output; // dont copy over just move pointer
    }

    return input;
}

double backward(double * pred_y, double * true_y, Model * model, char * loss_func, int batch_size) {
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

        double running_loss = 0.0;
        for(int idx = 0; idx < num_images; idx+=dataset_info->batch_size) {
            double * pred_y = forward(model, dataset, dataloader, idx, dataset_info->batch_size);
            double * true_y = load_labels(labels, dataloader, idx, dataset_info->batch_size);
            double loss = backward(pred_y, true_y, model, dataset_info->loss_func, dataset_info->batch_size);
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

    double accuracy = 0.0;
    for(int idx = 0; idx < num_images; idx+=dataset_info->batch_size) {
        double * pred_y = forward(model, dataset, dataloader, idx, dataset_info->batch_size);
        double * true_y = load_labels(labels, dataloader, idx, dataset_info->batch_size);
        
        for(int i = 0; i < dataset_info->batch_size; i++) {
            if(arg_max(pred_y,i) == arg_max(true_y,i))
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

    double * pred_y = forward(model, dataset, dataloader, 0, 1);
    double * true_y = load_labels(labels, dataloader, 0, 1);
    printf("Predicted: %d, Actual: %d\n\n", arg_max(pred_y,0), arg_max(true_y,0));

    close_dataset(dataset);
    close_dataset(labels);
}