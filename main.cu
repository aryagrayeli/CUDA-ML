#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "model.h"

char *program;
char *current;


void skip() {
    while (isspace(*current)) current += 1;
}

bool consume(const char *str) {
    skip();
    size_t i = 0;
    while (true) {
        char const expected = str[i];
        char const found = current[i];
        if (expected == 0) {
            /* survived to the end of the expected string */
            current += i;
            return true;
        }
        if (expected != found) return false;
        // assertion: found != 0
        i += 1;
    }
}

void fail() {
    printf("failed at offset %ld\n", (size_t)(current - program));
    printf("%s\n", current);
    exit(1);
}

void end_or_fail() {
    while (isspace(*current))
    {
        current += 1;
    }
    if (*current != 0)
        fail();
}

void consume_or_fail(const char *str) {
    if (!consume(str)) fail();
}

uint64_t consume_literal() {
  skip();

  bool negate = false;
  if(*current == '-') {
    current++;
    negate = true;
  }
  if (isdigit(*current)) {
    uint64_t v = 0;
    do {
      v = 10 * v + ((*current) - '0');
      current += 1;
    } while (isdigit(*current));
    return v * (negate ? -1 : 1);
  } else {
    fail();
    return 0;
  }
}

char* consume_identifier() {
  skip();

  if (isalpha(*current)) {
    char* start = current;
    do {
      current += 1;
    } while(isalnum(*current) || *current == '.' || *current == '_' || *current == '-');
    char* str = (char*)(malloc(sizeof(char)*(int)(current-start)));
    for(int i=0;i<current-start;i++) str[i] = start[i];
    return str;
  } 
  else {
    return NULL;
  }
}

int main(int argc, char **argv) {
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    // determine its size (std::filesystem::get_size?)
    struct stat file_stats;
    int rc = fstat(fd, &file_stats);
    if (rc != 0) {
        perror("fstat");
        exit(1);
    }

    // map the file in my address space
    char *prog = (char *)mmap(
        0,
        file_stats.st_size,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0);
    if (prog == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    program = prog;
    current = program;

    consume_or_fail("Layers: ");
    uint64_t layers = consume_literal();
    uint64_t * layers_size = (uint64_t *) malloc(sizeof(uint64_t) * layers);
    char ** activation_function = (char **) malloc(sizeof(char*) * layers);

    for(int i=0;i<layers;i++) {
        layers_size[i] = consume_literal();
        if(i != 0) {
            consume_or_fail("(");
            activation_function[i] = consume_identifier();
            consume_or_fail(")");
        }
        if(i != layers-1) consume_or_fail("->");
    }

    ArchInfo * arch_info = (ArchInfo *) malloc(sizeof(ArchInfo));
    arch_info->layers = layers;
    arch_info->layers_size = layers_size;
    arch_info->activation_function = activation_function;

    DatasetInfo * dataset_info = (DatasetInfo *) malloc(sizeof(DatasetInfo));
    consume_or_fail("Train: ");
    dataset_info->train_files = consume_identifier();

    consume_or_fail("Test: ");
    dataset_info->test_files = consume_identifier();
    
    consume_or_fail("Checkpoint: ");
    dataset_info->checkpoint_path = consume_identifier();

    consume_or_fail("Epochs: ");
    dataset_info->epochs = consume_literal();

    consume_or_fail("Batch Size: ");
    dataset_info->batch_size = consume_literal();

    consume_or_fail("Loss: ");
    dataset_info->loss_func = consume_identifier();

    printf("Loaded Arch File\n\n");

    Model * model = train(dataset_info, arch_info);
    // Model * model = load_model(dataset_info->checkpoint_path, arch_info); // to load model if not training

    test(model, dataset_info, arch_info);
    predict(model, dataset_info, arch_info, 1);

    // free everything in model, dataset_info and arch_info
    free(dataset_info->train_files);
    free(dataset_info->test_files);
    free(dataset_info->checkpoint_path);
    free(dataset_info->loss_func);
    free(dataset_info);

    for(int i = 0; i < arch_info->layers; i++)
      free(arch_info->activation_function[i]);
    free(arch_info->activation_function);
    free(arch_info->layers_size);

    for(int i = 0; i < arch_info->layers-1; i++) {
      cudaFree(model->weights[i]);
      cudaFree(model->biases[i]);
    }
    free(model->weights);
    free(model->biases);
    free(model->arch_info);
    free(model);
}