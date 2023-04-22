# CUDA-ML
Implementing an ML language in CUDA.

- Specify an ML language for feed forward NNs in a .arch file
- Create a C interpreter that can run the program in CUDA on a GPU / Use Cuda to interpret the .arch file and execute it on a GPU
    - Both train script and test script (different inputs to each)
        - Train: 
            - Inputs: .arch file, folder of train dataset (each entry is folder w/ inputs (.in) and outputs (.out) )
            - Outputs: .pth file w/ model weights + print out loss
        - Test:
            - Inputs: .arch file, test dataset (input data files (.in))
            - Outputs: output data file (.out)
- .arch format (must adhere to this format, otherwise undefined behavior):
    - Input layer size -> hidden layer size (hidden layer activation) -> … -> output layer size (output layer activation)
        - Activations: ReLU, sigmoid, tanh, softmax
    - Path to train dataset (Train: path)
    - Path to test dataset (Test: path)
    - Path to where weights while be saved (Checkpoint: path)
    - Number of training epochs (Epochs: #)
    - Batch size (Batch Size: #)
    - Learning Rate (Alpha: double)
    - L2 Regularization Parameter (Epsilon: double)
        - 0 means no regularization
    - Loss function (Loss: name)
        - Losses: MSE, CrossEntropy


Notes:
- Softmax can only be used as the final layer and only with CrossEntropy loss (not MSE!)
- Optimizer is currently Mini-Batch Gradient Descent

- How to compile: nvcc -rdc=true -lcudadevrt main.cu -o run.exe
- How to run: CUDA_VISIBLE_DEVICES=4 ./run.exe format.arch
- CUDA Guide: https://www.cs.utexas.edu/~rossbach/cs378h/papers/cuda-programming.pdf

- For reference, the format.arch file has an example that gets 98% test accuracy after 2 minutes of training
- For large networks, doing ReLU w/ a softmax causes exploding gradients
    - Might need to add gradient clipping since we already have good weight initialization and softmax max normalization


Features to Add:
- More Optimizers like Adam
- Convolutional Layers
- Normalization Layers
- Pooling Layers
- Dropout
- More Loss Functions
- More Activation Functions