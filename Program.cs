﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace BackPropNN;

interface Neuron
{
    double FeedForward(Vector inputs);
    Vector Backpropagate(Vector inputs, double error, double learningRate);
}

interface Layer
{
    Vector FeedForward(Vector inputs);
    Vector Backpropagate(Vector inputs, Vector errors, double learningRate);
}

interface NeuralNetwork
{
    Vector FeedForward(Vector inputs);
    void BackPropagate(Vector inputs, Vector errors, double learningRate);
}

interface NeuralNetworkFactory
{
    Neuron CreateNeuron(int inputSize);

    Layer CreateLayer(int neuronCount, int inputsPerNeuron);
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
        //XOR();
        MNist();
    }

    static void XOR()
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

    static void MNist() 
    {
        // Load MNIST data
        var (trainingInputs, trainingLabels) = MNistLoader.Load();
        
        Console.Out.WriteLine("Size of training set: " + trainingLabels.Length);
        Console.Out.WriteLine("Size of sample:" + trainingInputs[0].Length);

        // Create a neural network with 784 inputs (28x28 images), 100 hidden neurons, and 10 outputs (digits 0-9)
        var network = new NeuralNetworkImpl(new NeuralNetworkFactoryImpl(), 784, 50, 10);

        // Train the network
        new Trainer(network, trainingInputs, trainingLabels, 0.1, 100).Train();
    }
}