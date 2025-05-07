using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BackPropNN;

// All a neuron does is:
// - take inputs, apply weights to them, and return an output.
// - propagate errors back to the previous layer
interface Neuron
{
    double FeedForward(Vector inputs);
    Vector Backpropagate(Vector inputs, double error, double learningRate);
}

class MultiInputNeuron : Neuron
{
    private Vector _weights;
    private double _bias = 0;
    private double _lastOutput; // Store the last output for backpropagation

    public Vector Weights => _weights; // Allow access to weights
    public double Bias => _bias;

    private static readonly Random _random = new Random();

    public MultiInputNeuron(int inputSize)
    {
        _weights = new Vector(inputSize);
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = (_random.NextDouble() * 2 - 1) * 0.1; // Range (-0.1, 0.1)
        }
        _bias = (_random.NextDouble() * 2 - 1) * 0.1;
    }

    public double FeedForward(Vector inputs)
    {
        if (inputs.Length != _weights.Length)
            throw new ArgumentException("Number of inputs must match number of weights");

        _lastOutput = Sigmoid(Vector.DotProduct(_weights, inputs) + _bias);
        return _lastOutput;
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private double SigmoidDerivative(double output) => output * (1 - output);

    public Vector Backpropagate(Vector inputs, double error, double learningRate)
    {
        if (inputs.Length != _weights.Length)
            throw new ArgumentException("Number of inputs must match number of weights");

        double sigmoidDeriv = SigmoidDerivative(_lastOutput);
        double delta = error * sigmoidDeriv;

        // Calculate errors for the previous layer
        Vector previousLayerErrors = new Vector(inputs.Length);
        for (int i = 0; i < _weights.Length; i++)
        {
            previousLayerErrors[i] = delta * _weights[i];
        }

        // Update each weight
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] += learningRate * delta * inputs[i];
        }

        // Update bias
        _bias += learningRate * delta;

        return previousLayerErrors;
    }

    public override string ToString() => $"{_weights}, Bias: {_bias:F6}";
}
class NeuralNetwork
{
    private List<Layer> _layers = new List<Layer>();

    public NeuralNetwork(Func<int, Neuron> neuronFactory, params int[] layerSizes)
    {
        // Create layers based on the provided sizes
        for (int i = 1; i < layerSizes.Length; i++)
        {
            _layers.Add(new Layer(layerSizes[i], layerSizes[i - 1], neuronFactory));
        }
    }

    public Vector FeedForward(Vector inputs)
    {
        Vector currentOutputs = inputs;

        // Pass inputs through each layer
        foreach (var layer in _layers)
        {
            currentOutputs = layer.FeedForward(currentOutputs);
        }

        return currentOutputs;
    }

    public void Train(double[][] trainingInputs, double[][] expectedOutputs,
                      double learningRate, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;

            // Train on each input-output pair
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                // Forward pass
                var outputs = FeedForward(new Vector(trainingInputs[i]));

                // Calculate error
                var errors = new Vector(outputs.Length);
                for (int j = 0; j < outputs.Length; j++)
                {
                    errors[j] = expectedOutputs[i][j] - outputs[j];
                    totalError += Math.Abs(errors[j]);
                }

                // Backward pass (backpropagation)
                BackPropagate(new Vector(trainingInputs[i]), errors, learningRate);
            }

            if ((epoch + 1) % 100 == 0 || epoch == 0)
            {
                Console.WriteLine($"Epoch {epoch + 1}: Error = {totalError / trainingInputs.Length}");
            }
        }
    }

    private void BackPropagate(Vector initialInputs, Vector outputErrors, double learningRate)
    {
        var currentErrors = outputErrors;

        // Store all activations for backward pass.
        // allActivations[0] will be the initial network inputs.
        // allActivations[1] will be the outputs of _layers[0] (inputs to _layers[1]).
        // allActivations[k+1] will be the outputs of _layers[k] (inputs to _layers[k+1]).
        List<Vector> allActivations = new List<Vector> { initialInputs };

        // Forward pass to collect all layer INPUTS (outputs of previous layers)
        var currentLayerInputsForForwardPass = initialInputs;
        foreach (var layer in _layers)
        {
            currentLayerInputsForForwardPass = layer.FeedForward(currentLayerInputsForForwardPass);
            allActivations.Add(currentLayerInputsForForwardPass); // Add the OUTPUT of this layer
        }

        // Backward pass through layers in reverse order
        // For _layers[i], the input it received was allActivations[i]
        // The output it produced was allActivations[i+1]
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            // The inputs to _layers[i] during the forward pass were the activations
            // from the previous layer, which are stored in allActivations[i].
            var layerOriginalInputs = allActivations[i];
            currentErrors = _layers[i].Backpropagate(layerOriginalInputs, currentErrors, learningRate);
        }
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < _layers.Count; i++)
        {
            sb.AppendLine($"Layer {i + 1}:");
            sb.AppendLine(_layers[i].ToString());
        }
        return sb.ToString();
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
        NeuralNetwork network = new NeuralNetwork((n) => new MultiInputNeuron(n), 2, 10, 1);

        // XOR training data
        double[][] inputs = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };

        double[][] expectedOutputs = new double[][]
        {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };

        // Train the network
        network.Train(inputs, expectedOutputs, 0.1, 100000);

        // Test the network
        Console.WriteLine("\nTesting the network:");
        foreach (var input in inputs)
        {
            var output = network.FeedForward(new Vector(input));
            Console.WriteLine($"Input: ({input[0]}, {input[1]}) -> Output: {output[0]:F4}");
        }
    }
}