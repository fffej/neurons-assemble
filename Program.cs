using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BackPropNN;

interface Layer
{
    Vector FeedForward(Vector inputs);
    Vector Backpropagate(Vector inputs, Vector errors, double learningRate);
}

interface Neuron
{
    double FeedForward(Vector inputs);
    Vector Backpropagate(Vector inputs, double error, double learningRate);
}

interface NeuralNetwork
{
    Vector FeedForward(Vector inputs);
    void BackPropagate(Vector initialInputs, Vector outputErrors, double learningRate);
}

interface NeuralNetworkFactory
{
    Neuron CreateNeuron(int inputSize);

    Layer CreateLayer(int neuronCount, int inputsPerNeuron);
}

class Trainer
{
    private NeuralNetwork _network;
    private double[][] _trainingInputs;
    private double[][] _expectedOutputs;
    private double _learningRate;
    private int _epochs;

    public Trainer(NeuralNetwork network, double[][] trainingInputs, double[][] expectedOutputs, double learningRate, int epochs)
    {
        _network = network;
        _trainingInputs = trainingInputs;
        _expectedOutputs = expectedOutputs;
        _learningRate = learningRate;
        _epochs = epochs;
    }

    public void Train()
    {
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            double totalError = 0;

            // Train on each input-output pair
            for (int i = 0; i < _trainingInputs.Length; i++)
            {
                // Forward pass
                var outputs = _network.FeedForward(new Vector(_trainingInputs[i]));

                // Calculate error
                var errors = new Vector(outputs.Length);
                for (int j = 0; j < outputs.Length; j++)
                {
                    errors[j] = _expectedOutputs[i][j] - outputs[j];
                    totalError += Math.Abs(errors[j]);
                }

                // Backward pass (backpropagation)
                _network.BackPropagate(new Vector(_trainingInputs[i]), errors, _learningRate);
            }

            if ((epoch + 1) % 100 == 0 || epoch == 0)
            {
                Console.WriteLine($"Epoch {epoch + 1}: Error = {totalError / _trainingInputs.Length}");
            }
        }
    }
}

class Program
{
    class NeuralNetworkFactoryImpl : NeuralNetworkFactory
    {
        public Neuron CreateNeuron(int inputSize) => new MultiInputNeuron(inputSize);
        public Layer CreateLayer(int neuronCount, int inputsPerNeuron) => new LayerImpl(neuronCount, inputsPerNeuron, CreateNeuron);
    }

    static void Main(string[] args)
    {
        // Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
        var network = new NeuralNetworkImpl(new NeuralNetworkFactoryImpl(), 2, 3, 1);

        // XOR training data
        double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] expectedOutputs = [ [0], [1], [1],  [0] ];

        new Trainer(network, inputs, expectedOutputs, 0.1, 100000).Train();
        Console.WriteLine("Training complete. Final weights and biases:");

        // Test the network
        Console.WriteLine("\nTesting the network:");
        foreach (var input in inputs)
        {
            var output = network.FeedForward(new Vector(input));
            Console.WriteLine($"Input: ({input[0]}, {input[1]}) -> Output: {output[0]:F4}");
        }
    }
}