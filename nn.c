#include <stdio.h>
#include <stdlib.h>
#include <math.h>  

struct neuron {
    int nIn;        // int number of inputs
    float* in;      // pointer to an array of inputs (chaged from int)
    float* w;       // pointer to an array of float weights
    float b;        // float bias
    float output;   // float output
};

struct layer {
    int nNeuron;            // in number of neurons in layer
    struct neuron* neurons; // pointer to an array of neurons
};

struct nn {
    int nLayers;            // number of layers in network
    struct layer* layers;   // pointer to an array of layers
};

// Function to create a new neuron
struct neuron createNeuron(int nIn) {               // creates a new neuron with int nin number of inputs
    struct neuron n;                                // creates a new neuron struct 'n'
    n.nIn = nIn;                                    // sets attribute for number of inputs
    n.in = (float*)malloc(nIn * sizeof(float));     // allocates memory for nIn float-sized chunks, to store the inputs
    n.w = (float*)malloc(nIn * sizeof(float));      // as above, to store the weights
    n.b = 0.0f;                                     // sets bias to 0.0
    return n;                                       // returns the new neuron n
}

// Function to create a new layer
struct layer createLayer(int nNeuron, int nIn) {                            // creates a new layer with nNeuron neurons, each taking nIn inputs
    struct layer l;                                                         // creates a new layer struct 'l'
    l.nNeuron = nNeuron;                                                    // sets the number of neurons to nNeuron
    l.neurons = (struct neuron*)malloc(nNeuron * sizeof(struct neuron));    // allocates memory for no. of neurons x size of neuron struct
    for (int i = 0; i < nNeuron; i++) {                                     // for loop that loops nNeuron times
        l.neurons[i] = createNeuron(nIn);                                       // and creates a new neuron in l.neurons[i] on each iteration
    }
    return l;                                                               // returns the new layer 'l'
}

// Function to create a new neural network
// creates a new NN with nLayers layers, and pointers to nNeuronsPerLayer and nInputsPerNeuron
struct nn createNN(int nLayers, int* nNeuronsPerLayer, int* nInputsPerNeuron) {     // the latter 2 args are arrays carrying values for each layer
    struct nn network;                                                              // creates a new nn struct 'network'
    network.nLayers = nLayers;                                                      // sets the number of layers attribute
    network.layers = (struct layer*)malloc(nLayers * sizeof(struct layer));         // allocates memory for number of layers x size of layer struct
    for (int i = 0; i < nLayers; i++) {                                             // for loop that loops nLayers times
                                                                                        // and creates a new layer on each iteration
        network.layers[i] = createLayer(nNeuronsPerLayer[i], nInputsPerNeuron[i]);      // using the nNeurons and nInputs at i in each array arg
    }
    return network;                                                                 // returns the new nn 'network'
}

// Function to free the dynamically allocated memory
void freeNN(struct nn network) {                                // frees memory from a given nn type struct
    for (int i = 0; i < network.nLayers; i++) {                 // for loop through each layer i in the struct
        for (int j = 0; j < network.layers[i].nNeuron; j++) {   // for loop through each neuron j in the layer
            free(network.layers[i].neurons[j].in);              // frees memory allocated to inputs for neuron j at layer i
            free(network.layers[i].neurons[j].w);               // as above, for weights
        }
        free(network.layers[i].neurons);                        // as above, for each neuron in the layer
    }
    free(network.layers);                                       // as above, for each layer in the network
}

// Activation function (sigmoid)
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));    // standard sigmoid activation on a float input x
}

// Forward propagation
// takes network, type struct nn, input, type array of floats, and output, type array of floats
void forwardPropagation(struct nn network, float* input, float* output) {
    for (int i = 0; i < network.nLayers; i++) {                             // for each layer in network
        for (int j = 0; j < network.layers[i].nNeuron; j++) {               // for each neuron in layer
            float sum = network.layers[i].neurons[j].b;                     // initialize sum equal to bias, then
            for (int k = 0; k < network.layers[i].neurons[j].nIn; k++) {    // for each input to neuron
                // calculate weight * input and add result to sum
                // right hand of multiplication is a ternary if/else that uses input[k] in the first layer
                // else, the output of all neurons in the previous layer
                sum += network.layers[i].neurons[j].w[k] * (i == 0 ? input[k] : network.layers[i-1].neurons[k].output);                
            }
            network.layers[i].neurons[j].output = sigmoid(sum);             // assign neuron output value = sigmoid activation of sum
        }
    }
    // for loop below copies output values of the neurons in the last layer to 'output' array
    // iterates over each neuron in the last layer (from 0 to network.layers[network.nLayers-1].nNeuron - 1)

    // network.nLayers-1 gives the index of the last layer in the network 
    // network.layers[network.nLayers-1].nNeuron gives the number of neurons in the last layer
    for (int i = 0; i < network.layers[network.nLayers-1].nNeuron; i++) {   // for each neuron in the last layer, copies output value of the neuron 
        output[i] = network.layers[network.nLayers-1].neurons[i].output;    // (network.layers[network.nLayers-1].neurons[i].output) to the corresponding index in the output array (output[i])
    }
}

int main() {
    int nLayers = 3;                        // set number of layers for network
    int nNeuronsPerLayer[] = {5, 3, 1};     // set number of neurons in each layer (no. of elements must match nLayers)
    int nInputsPerNeuron[] = {2, 5, 3};     // set number of inputs per neuron in each layer (dependent on inputs and neuron in prv layer)

    struct nn network = createNN(nLayers, nNeuronsPerLayer, nInputsPerNeuron); // use createNN function with above parameters to create a new NN

    // Use the neural network

    // set random weights and biases
    for (int i = 0; i < network.nLayers; i++) {                                 // for each layer
        for (int j = 0; j < network.layers[i].nNeuron; j++) {                   // for each neuron in the layer
            for (int k = 0; k < network.layers[i].neurons[j].nIn; k++) {        // for each input to the neuron
                network.layers[i].neurons[j].w[k] = (float)rand() / RAND_MAX;   // set random weight
            }
            network.layers[i].neurons[j].b = (float)rand() / RAND_MAX;          // set random bias
        }
    }

    // provides arrays for inputs (initiliazed with values) and outputs (to catch outputs)
    float input[] = {0.5f, 0.8f};           // provide inputs (no. of inputs muct match inputs to first layer in nInputsPerNeuron)
    float output[1];                        // size 1 array to hold output (size must match no. neurons in final layer)

    forwardPropagation(network, input, output);     // run forwardPropagation on the network
    printf("Output: %f\n", output[0]);              // print the output

    freeNN(network);                        // use freeNN function to free memory allocated while running the program

    return 0;                               // return 0 if program executes successfully
}
