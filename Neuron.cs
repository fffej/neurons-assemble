using System.Text;

namespace BackPropNN;

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