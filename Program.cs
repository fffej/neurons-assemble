using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using BenchmarkDotNet.Running;

namespace BackPropNN;

interface Neuron
{
    double FeedForward(Vector inputs);
    void Backpropagate(Vector inputs, double error, Vector errors, double learningRate);
}

interface Layer
{
    Vector FeedForward(Vector inputs);
    void Backpropagate(Vector inputs, Vector errors, Vector outputErrors, double learningRate);
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
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("  dotnet run bench    - Run benchmarks");
            Console.WriteLine("  dotnet run or       - Run OR demo");
            Console.WriteLine("  dotnet run xor      - Run XOR demo");
            Console.WriteLine("  dotnet run xor-opt  - Run XOR demo with optimized layers");
            Console.WriteLine("  dotnet run mnist    - Run MNIST demo (100 epochs)");
            return;
        }

        switch (args[0].ToLower())
        {
            case "bench":
                BenchmarkRunner.Run<TrainingBenchmark>();
                break;
            case "or":
                OR();
                break;
            case "xor":
                XOR();
                break;
            case "xor-opt":
                XOROptimized();
                break;
            case "mnist":
                MNist(1);
                break;
            default:
                Console.WriteLine($"Unknown command: {args[0]}");
                Console.WriteLine("Available commands: bench, or, xor, xor-opt, mnist");
                break;
        }
    }

    class OrNeuron
    {
        private double _w1 = 0;
        private double _w2 = 0;
        private double _bias = 0;

        public double FeedForward(double x1, double x2) => Sigmoid(_w1 * x1 + _w2 * x2 + _bias);

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        private double SigmoidDerivative(double output) => output * (1 - output);

        public void Backpropagate(double x1, double x2, double error, double learningRate)
        {
            double output = FeedForward(x1, x2);
            double sigmoidDeriv = SigmoidDerivative(output);
            
            _w1 += learningRate * error * sigmoidDeriv * x1;
            _w2 += learningRate * error * sigmoidDeriv * x2;
            _bias += learningRate * error * sigmoidDeriv;
        }
    }

    static void OR()
    {
        double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] expectedOutputs = [ [0], [1], [1],  [1] ];

        var neuron = new OrNeuron();

        for (int epoch = 0; epoch < 100000; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double output = neuron.FeedForward(inputs[i][0], inputs[i][1]);
                double error = expectedOutputs[i][0] - output;
                neuron.Backpropagate(inputs[i][0], inputs[i][1], error, 0.1);
                totalError += Math.Abs(error);
            }
            if (epoch % 1000 == 0) 
            {
                Console.WriteLine($"Epoch {epoch}: Total error = {totalError}");
            }
        }

        Console.WriteLine("Testing the network:");
        foreach (var input in inputs)
        {
            double output = neuron.FeedForward(input[0], input[1]);
            Console.WriteLine($"Input: ({input[0]}, {input[1]}) -> Output: {output:F4}");
        }
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
            var output = network.FeedForward(Vector.FromSpan(input.AsSpan()));
            Console.WriteLine($"Input: ({input[0]}, {input[1]}) -> Output: {output[0]:F4}");
        }
    }

    static void XOROptimized()
    {
        Console.WriteLine("Running XOR with Optimized Layers (Structure-of-Arrays)");
        
        // Create a neural network with 2 inputs, 3 hidden neurons, and 1 output using optimized layers
        var network = new NeuralNetworkImpl(new OptimizedNeuralNetworkFactory(), 2, 3, 1);

        // XOR training data
        double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] expectedOutputs = [ [0], [1], [1],  [0] ];

        new Trainer(network, inputs, expectedOutputs, 0.1, 100000).Train();
        Console.WriteLine("Training complete. Final weights and biases:");

        // Test the network
        Console.WriteLine("\nTesting the optimized network:");
        foreach (var input in inputs)
        {
            var output = network.FeedForward(Vector.FromSpan(input.AsSpan()));
            Console.WriteLine($"Input: ({input[0]}, {input[1]}) -> Output: {output[0]:F4}");
        }
    }

    static void MNist(int epochs = 100) 
    {
        // Load MNIST data
        var (trainingInputs, trainingLabels) = MNistLoader.Load();
        
        Console.Out.WriteLine("Size of training set: " + trainingLabels.Length);
        Console.Out.WriteLine("Size of sample:" + trainingInputs[0].Length);

        // Create a neural network with 784 inputs (28x28 images), 50 hidden neurons, and 10 outputs (digits 0-9)
        var network = new NeuralNetworkImpl(new NeuralNetworkFactoryImpl(), 784, 50, 10);

        // Train the network
        new Trainer(network, trainingInputs, trainingLabels, 0.1, epochs).Train();
    }
}