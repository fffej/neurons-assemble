using System.Text;

namespace BackPropNN;

public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble(); // Uniform(0,1] random
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }
}

class MultiInputNeuron : Neuron
{
    private Vector _weights;
    private double _bias = 0;
    // This just avoids having to recalculate the output
    private double _lastOutput; 

    private static readonly Random _random = new Random();

    public MultiInputNeuron(int inputSize)
    {
        // Xavier initialization
        _weights = new Vector(inputSize);
        double sqrtVariance = Math.Sqrt(1.0 / inputSize);
        for (int i=0; i < _weights.Length; i++)
            _weights[i] = _random.NextGaussian(0, sqrtVariance);

        _bias = _random.NextGaussian(0, sqrtVariance);
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
        Vector errors = delta * _weights;

        // Update weights and bias
        _weights += learningRate * delta * inputs;        
        _bias += learningRate * delta;
        
        return errors;
    }

    public override string ToString() => $"{_weights}, Bias: {_bias:F6}";
}