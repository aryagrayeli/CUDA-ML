#pragma once

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "utils.h"
#include "matmul.cu"

#define ALPHA (0.01)
#define THREADS (32)
#define NUM_TRAIN (60000)
#define NUM_TEST (10000)
#define IM_WIDTH (28)
#define IM_HEIGHT (28)

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
        char * act_func = arch_info->activation_function[i+1];

        dim3 gridSz((n*m + THREADS - 1) / THREADS, 1, 1);
        dim3 blockSz(THREADS, 1, 1);

        curandState * state;
        cudaMalloc(&state, n*m * sizeof(curandState));
        setup_kernel<<<gridSz,blockSz>>>(state, time(NULL));

        if(strcmp(act_func, "ReLU") == 0)
            he_init<<<gridSz, blockSz>>>(model->weights[i], state, sqrt(2.0/n), (int) n*m); // values sampled from G(0.0, sqrt(2/n))
        else
            xavier_init<<<gridSz, blockSz>>>(model->weights[i], state, sqrt(6.0/(n+m)), (int) n*m); // values sampled from U(-sqrt(6/(n+m)), sqrt(6/(n+m)))

        cudaFree(state);
    }

    model->biases = (double **) malloc(sizeof(double*) * (arch_info->layers-1));
    for(int i = 0; i < arch_info->layers-1; i++) {
        int m = (int) arch_info->layers_size[i+1];
        cudaMalloc(&(model->biases[i]), sizeof(double) * m);

        dim3 gridSz((m + THREADS - 1) / THREADS, 1, 1);
        dim3 blockSz(THREADS, 1, 1);

        // gpu initialize model->biases[i] in parallel
        zero_init<<<gridSz, blockSz>>>(model->biases[i], m);
    }

    model->arch_info = arch_info;
    return model;
}

void save_model(Model * model, char * checkpoint_path) {
    printf("Saving Model\n");

    FILE* fp = fopen(checkpoint_path, "w");// "w" means that we are going to write on this file

    cudaDeviceSynchronize();

    fprintf(fp, "Biases:\n");
    for(int i=0;i<model->arch_info->layers-1;i++) {
        uint64_t N = model->arch_info->layers_size[i+1];

        double * bias = (double *) malloc(sizeof(double) * N);
        cudaMemcpy(bias, model->biases[i], sizeof(double) * N, cudaMemcpyDeviceToHost);

        for(int j = 0; j < N; j++)
            fprintf(fp, "%lf ", bias[j]);
        fprintf(fp, "\n");
    }

    fprintf(fp, "Weights:\n");
    for(int i=0;i<model->arch_info->layers-1;i++) {
        uint64_t N = model->arch_info->layers_size[i+1];
        uint64_t M = model->arch_info->layers_size[i];

        double * weight = (double *) malloc(sizeof(double) * N * M);
        cudaMemcpy(weight, model->weights[i], sizeof(double) * N * M, cudaMemcpyDeviceToHost);

        for(int j = 0; j < N*M; j++)
            fprintf(fp, "%lf ", weight[j]);
        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Model Saved\n");
}

Model * load_model(char * checkpoint_path, ArchInfo * arch_info) {
    printf("Loading Model\n");

    FILE * fp = fopen(checkpoint_path, "r");

    Model * model = (Model *) malloc(sizeof(Model));
    model->arch_info = arch_info;
    model->biases = (double **) malloc(sizeof(double*) * (arch_info->layers-1));
    model->weights = (double **) malloc(sizeof(double*) * (arch_info->layers-1));

    fscanf(fp, "Biases:\n");
    for(int i=0;i<arch_info->layers-1;i++) {
        uint64_t N = arch_info->layers_size[i+1];

        double * bias = (double *) malloc(sizeof(double) * N);
        for(int j = 0; j < N; j++)
            fscanf(fp, "%lf ", &(bias[j]));
        fscanf(fp,"\n");

        cudaMalloc(&(model->biases[i]), sizeof(double) * N);
        cudaMemcpy(model->biases[i], bias, sizeof(double) * N, cudaMemcpyHostToDevice);
    }

    fscanf(fp, "Weights:\n");
    for(int i=0;i<arch_info->layers-1;i++) {
        uint64_t N = arch_info->layers_size[i+1];
        uint64_t M = arch_info->layers_size[i];

        double * weight = (double *) malloc(sizeof(double) * N * M);
        for(int j = 0; j < N*M; j++)
            fscanf(fp, "%lf ", &(weight[j]));
        fscanf(fp,"\n");
        
        cudaMalloc(&(model->weights[i]), sizeof(double) * N * M);
        cudaMemcpy(model->weights[i], weight, sizeof(double) * N * M, cudaMemcpyHostToDevice);
    }

    fclose(fp);

    printf("Model Loaded\n");
    return model;
}

double ** forward(Model * model, FILE * dataset, int * dataloader, int idx, int batch_size) {
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
        free(pixels);
    }

    double ** layer_vecs = (double **) malloc(sizeof(double*) * arch_info->layers);

    for(int l = 0; l < arch_info->layers-1; l++) {
        int prev_size = (int) arch_info->layers_size[l];
        int next_size = (int) arch_info->layers_size[l+1];

        double * output;
        cudaMalloc(&output, sizeof(double) * next_size * batch_size);

        dim3 gridSz((batch_size + THREADS - 1)/THREADS,1,1);
        dim3 blockSz(THREADS,1,1);

        batch_matrix_mul<<<gridSz,blockSz>>>(weights[l], input, output, next_size, prev_size, batch_size);
        batch_vector_add<<<gridSz,blockSz>>>(output, biases[l], output, next_size, batch_size);
        
        char * act = arch_info->activation_function[l+1];
        if(strcmp(act, "ReLU") == 0)
            batch_vector_relu<<<gridSz,blockSz>>>(output, output, next_size, batch_size);
        if(strcmp(act, "sigmoid") == 0)
            batch_vector_sigmoid<<<gridSz,blockSz>>>(output, output, next_size, batch_size);
        if(strcmp(act, "tanh") == 0)
            batch_vector_tanh<<<gridSz,blockSz>>>(output, output, next_size, batch_size);
        if(strcmp(act, "softmax") == 0)
            batch_vector_softmax<<<gridSz,blockSz>>>(output, output, next_size, batch_size);

        layer_vecs[l] = input;
        input = output; // dont copy over just move pointer
    }
    layer_vecs[arch_info->layers-1] = input;

    return layer_vecs;
}

double backward(double ** layer_vecs, double * true_y_cpu, Model * model, char * loss_func, int batch_size) {
    double ** weights = model->weights;
    double ** biases = model->biases;
    ArchInfo * arch_info = model->arch_info;
    uint64_t num_layers = arch_info->layers;
    uint64_t * layers_size = arch_info->layers_size;

    // get last layer output
    double * pred_y = layer_vecs[num_layers-1];
    
    // copy true_y to gpu mem
    double * true_y;
    cudaMalloc(&true_y, sizeof(double) * num_classes * batch_size);
    cudaMemcpy(true_y, true_y_cpu, sizeof(double) * num_classes * batch_size, cudaMemcpyHostToDevice);

    double * batch_loss;
    cudaMalloc(&batch_loss, sizeof(double) * batch_size);

    if(strcmp(loss_func, "MSE") == 0) {
        dim3 gridSz((batch_size + THREADS - 1)/THREADS,1,1);
        dim3 blockSz(THREADS,1,1);

        mse_loss<<<gridSz,blockSz>>>(pred_y, true_y, batch_loss, layers_size[num_layers-1], batch_size);
    }
    else // invalid, only MSE loss supported currently
        return -1.0;
    
    double ** layer_delts = (double **) malloc(sizeof(double*) * (num_layers-1));

    for(int i = num_layers-2; i >= 0; i--) {
        double * delta;
        cudaMalloc(&delta, sizeof(double) * layers_size[i+1] * batch_size);

        if(i == num_layers-2) {
            long totalThreads = layers_size[i+1] * batch_size;
            dim3 gridSz((totalThreads + THREADS - 1)/THREADS, 1, 1);
            dim3 blockSz(THREADS, 1, 1);
            vector_sub<<<gridSz, blockSz>>>(layer_vecs[i+1], true_y, delta, layers_size[i+1] * batch_size);
        } 
        else {
            double * weight_transpose;
            cudaMalloc(&weight_transpose, sizeof(double) * layers_size[i+2] * layers_size[i+1]);
            dim3 gridSz2((layers_size[i+2] + THREADS - 1)/THREADS, (layers_size[i+1] + THREADS - 1)/THREADS, 1);
            dim3 blockSz2(THREADS, THREADS);
            matrix_trans<<<gridSz2, blockSz2>>>(weights[i+1], weight_transpose, layers_size[i+2], layers_size[i+1]);

            batch_matrix_mul<<<1, batch_size>>>(weight_transpose, layer_delts[i+1], delta, layers_size[i+1], layers_size[i+2], batch_size);
            cudaFree(weight_transpose);
        }

        // double * delta_cpu = (double *) malloc(sizeof(double) * layers_size[i+1] * batch_size);
        // cudaDeviceSynchronize();
        // cudaMemcpy(delta_cpu, delta, sizeof(double) * layers_size[i+1] * batch_size, cudaMemcpyDeviceToHost);
        // for(int j = 0; j < layers_size[i+1] * batch_size; j++)
        //     printf("%lf ", delta_cpu[j]);
        // printf("\n\n");


        double * d_act;
        cudaMalloc(&d_act, sizeof(double) * layers_size[i+1] * batch_size);
        batch_matrix_mul<<<1, batch_size>>>(weights[i], layer_vecs[i], d_act, layers_size[i+1], layers_size[i], batch_size);
        batch_vector_dsigmoid<<<1, batch_size>>>(d_act, d_act, layers_size[i+1], batch_size);

        dim3 gridSz((layers_size[i+1] * batch_size + THREADS - 1)/THREADS, 1, 1);
        dim3 blockSz(THREADS, 1, 1);
        vector_hadamard<<<gridSz, blockSz>>>(delta, d_act, delta, layers_size[i+1] * batch_size);

        cudaFree(d_act);
        layer_delts[i] = delta;
    }

    // update weights and biases
    for(int i = 0; i < num_layers-1; i++) {
        double * gradients;
        cudaMalloc(&gradients, sizeof(double) * layers_size[i+1] * layers_size[i]);
        dim3 gridSz1((layers_size[i+1] * layers_size[i] + THREADS - 1)/THREADS, 1, 1);
        dim3 blockSz1(THREADS, 1, 1);
        zero_init<<<gridSz1, blockSz1>>>(gradients, layers_size[i+1] * layers_size[i]);

        dim3 gridSz2((layers_size[i+1] + THREADS - 1)/THREADS, (layers_size[i] + THREADS - 1)/THREADS, 1);
        dim3 blockSz2(THREADS, THREADS);
        vector_op<<<gridSz2, blockSz2>>>(layer_delts[i], layer_vecs[i], gradients, layers_size[i+1], layers_size[i], batch_size);
        matrix_sub_scalar<<<gridSz2, blockSz2>>>(weights[i], gradients, (double) ALPHA, weights[i], layers_size[i+1], layers_size[i]);

        vector_sub_scalar<<<1, layers_size[i+1]>>>(biases[i], layer_delts[i], (double) ALPHA, biases[i], layers_size[i+1]);

        cudaFree(gradients);
    }

    // output summed loss
    double * loss = (double *) malloc(sizeof(double) * batch_size);
    cudaDeviceSynchronize();
    cudaMemcpy(loss, batch_loss, sizeof(double) * batch_size, cudaMemcpyDeviceToHost);

    double sum_loss = 0.0;
    for(int i = 0; i < batch_size; i++)
        sum_loss += loss[i];


    // free everything
    for(int i = 0; i < arch_info->layers; i++) {
        cudaFree(layer_vecs[i]);
        cudaFree(layer_delts[i]);
    }

    free(layer_vecs);
    free(layer_delts);
    free(true_y_cpu);
    cudaFree(true_y);
    cudaFree(batch_loss);
    free(loss);

    return sum_loss; // loss value
}


Model * train(DatasetInfo * dataset_info, ArchInfo * arch_info) {
    printf("Beginning Training\n");

    FILE * dataset = load_dataset(concat(dataset_info->train_files,"/train-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info->train_files,"/train-labels.idx1-ubyte"));
    int32_t num_images = NUM_TRAIN;

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
            double ** layer_vecs = forward(model, dataset, dataloader, idx, dataset_info->batch_size);
            double * true_y = load_labels(labels, dataloader, idx, dataset_info->batch_size);
            double loss = backward(layer_vecs, true_y, model, dataset_info->loss_func, dataset_info->batch_size);
            running_loss+=loss;
        }
        running_loss /= num_images;
        printf("After Epoch %d, Train Loss: %f\n", e, running_loss);
    }

    free(dataloader);

    save_model(model, dataset_info->checkpoint_path);
    close_dataset(dataset);
    close_dataset(labels);

    printf("\n");

    return model;
}

void test(Model * model, DatasetInfo * dataset_info, ArchInfo * arch_info) {
    printf("Beginning Testing\n");

    FILE * dataset = load_dataset(concat(dataset_info->test_files,"/test-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info->test_files,"/test-labels.idx1-ubyte"));
    int32_t num_images = NUM_TEST;

    int * dataloader = (int *) malloc(sizeof(int) * num_images);
    for(int i = 0; i < num_images; i++) {
        dataloader[i] = i;
    }

    printf("Dataset and Model Initalized\n");

    double accuracy = 0.0;
    for(int idx = 0; idx < num_images; idx+=dataset_info->batch_size) {
        double ** layer_vecs = forward(model, dataset, dataloader, idx, dataset_info->batch_size);
        double * true_y = load_labels(labels, dataloader, idx, dataset_info->batch_size);

        double * pred_y = (double *) malloc(sizeof(double) * num_classes * dataset_info->batch_size);
        cudaDeviceSynchronize();
        cudaMemcpy(pred_y, layer_vecs[arch_info->layers-1], sizeof(double) * num_classes * dataset_info->batch_size, cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < dataset_info->batch_size; i++) {
            if(arg_max(pred_y,i) == arg_max(true_y,i))
                accuracy++;
        }

        // free
        for(int i = 0; i < arch_info->layers; i++)
            cudaFree(layer_vecs[i]);

        free(layer_vecs);
        free(pred_y);
        free(true_y);
    }

    free(dataloader);

    accuracy = (100.0 * accuracy) / num_images;
    printf("Test Accuracy of %f percent for %d images\n\n", accuracy, num_images);

    close_dataset(dataset);
    close_dataset(labels);
}

void predict(Model * model, DatasetInfo * dataset_info, ArchInfo * arch_info, int image_idx) {
    printf("Beginning Prediction on Image #%d\n", image_idx);
    
    FILE * dataset = load_dataset(concat(dataset_info->test_files,"/test-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info->test_files,"/test-labels.idx1-ubyte"));

    int dataloader[1];
    dataloader[0] = image_idx;

    printf("Image and Model Initalized\n");

    double ** layer_vecs = forward(model, dataset, dataloader, 0, 1);
    double * true_y = load_labels(labels, dataloader, 0, 1);

    double * pred_y = (double *) malloc(sizeof(double) * num_classes * 1);
    cudaDeviceSynchronize();
    cudaMemcpy(pred_y, layer_vecs[arch_info->layers-1], sizeof(double) * num_classes * 1, cudaMemcpyDeviceToHost);

    double * image = get_image(dataset, image_idx);

    printf("Displaying Image #%d:\n", image_idx);
    for(int i = 0; i < IM_WIDTH; i++) {
        for(int j = 0; j < IM_HEIGHT; j++) {
            if(image[IM_HEIGHT*i+j] > 0.01)
                printf(". ");
            else
                printf("  ");
        }
        printf("\n");
    }
    printf("\n");

    printf("Output Vector: ");
    for(int i = 0; i < num_classes; i++) {
        printf("%lf ", pred_y[i]);
    }
    printf("\n");
 
    printf("Predicted: %d, Actual: %d\n\n", arg_max(pred_y,0), arg_max(true_y,0));

    // free
    for(int i = 0; i < arch_info->layers; i++)
        cudaFree(layer_vecs[i]);

    free(layer_vecs);
    free(pred_y);
    free(true_y);

    close_dataset(dataset);
    close_dataset(labels);
}
