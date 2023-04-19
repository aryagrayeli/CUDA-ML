#include <stdio.h>
#include <stdint.h>

#include <main.c>
#include <utils.cu>


typedef struct Model {
    float *** weights; // list of 2D weight matrix for each layer
    float ** biases; // list of 1D bias vectors for each layer
    char** activation_function; // list of activation functions for each layer
} Model;

Model * initialize_model(ArchInfo arch_info) {
    Model * model = (Model *) malloc(sizeof(Model));

    model->weights = NULL; // TODO
    model->biases = NULL; // TODO

    model->activation_function = arch_info.activation_function;
    return model;
}

void save_model(Model * model, char * checkpoint_path) {
    // TODO
}

Model * load_model(char * checkpoint_path) {
    // TODO
}

int * forward(Model * model, FILE * dataset, int * dataloader, int idx, int batch_size) {
    // TODO
}

void backward(int * pred_y, int * true_y, Model * model, char * loss_func, int batch_size) {
    // TODO
}


void train(DatasetInfo dataset_info, ArchInfo arch_info) {
    FILE * dataset = load_dataset(concat(dataset_info.train_files,"/train-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info.train_files,"/train-labels.idx1-ubyte"));
    int32_t num_images = get_dataset_size(dataset);

    int * dataloader = malloc(sizeof(int) * num_images);
    for(int i = 0; i < num_images; i++) {
        dataloader[i] = i;
    }

    Model * model = initialize_model(arch_info);

    for(int e = 0; e < dataset_info.epochs; e++) {
        shuffle(dataloader, num_images);

        for(int idx = 0; idx < num_images; idx+=dataset_info.batch_size) {
            int * pred_y = forward(model, dataset, dataloader, idx, dataset_info.batch_size);
            int * true_y = load_labels(labels, dataloader, idx, dataset_info.batch_size);
            backward(pred_y, true_y, model, dataset_info.loss_func, dataset_info.batch_size);
        }
    }

    save_model(model, dataset_info.checkpoint_path);
    close_dataset(dataset);
    close_dataset(labels);
}

void test(DatasetInfo dataset_info, ArchInfo arch_info) {
    FILE * dataset = load_dataset(concat(dataset_info.test_files,"/test-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info.train_files,"/test-labels.idx1-ubyte"));
    int32_t num_images = get_dataset_size(dataset);
    
    Model * model = load_model(dataset_info.checkpoint_path);

    int * dataloader = malloc(sizeof(int) * num_images);
    for(int i = 0; i < num_images; i++) {
        dataloader[i] = i;
    }

    float accuracy = 0.0;
    for(int idx = 0; idx < num_images; idx+=dataset_info.batch_size) {
        int * pred_y = forward(model, dataset, dataloader, idx, dataset_info.batch_size);
        int * true_y = load_labels(labels, dataloader, idx, dataset_info.batch_size);
        
        for(int i = 0; i < dataset_info.batch_size; i++) {
            if(pred_y[i] == true_y[i])
                accuracy++;
        }
    }

    accuracy/=num_images;
    // print(accuracy out of num_images test samples)

    close_dataset(dataset);
    close_dataset(labels);
}

void predict(DatasetInfo dataset_info, ArchInfo arch_info, int image_idx) {
    FILE * dataset = load_dataset(concat(dataset_info.test_files,"/test-images.idx3-ubyte"));
    FILE * labels = load_dataset(concat(dataset_info.train_files,"/test-labels.idx1-ubyte"));

    Model * model = load_model(dataset_info.checkpoint_path);

    int dataloader[1];
    dataloader[0] = image_idx;

    int * pred_y = forward(model, dataset, dataloader, 0, 1);
    int * true_y = load_labels(labels, dataloader, 0, 1);
    // print(pred_y[0], true_y[0])

    close_dataset(dataset);
    close_dataset(labels);
}