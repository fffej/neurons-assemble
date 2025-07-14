using System.Text;
using System.Runtime.CompilerServices;

namespace BackPropNN;

class LayerImpl : Layer
{
    private Neuron[] _neurons;
    private Vector _output;
    private Vector _neuronErrors; // Pre-allocated buffer for neuron errors

    public LayerImpl(int neuronCount, int inputsPerNeuron, Func<int, Neuron> funcFactory)
    {
        _neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            _neurons[i] = funcFactory(inputsPerNeuron); 
        _output = new Vector(_neurons.Length);
        _neuronErrors = new Vector(inputsPerNeuron); // Pre-allocate buffer for neuron errors
    }

    public Vector FeedForward(Vector inputs)
    {
        var outputSpan = _output.AsSpan();
        for (int i = 0; i < _neurons.Length; i++)
        {
            outputSpan[i] = _neurons[i].FeedForward(inputs);
        }
        return _output;
    }

    public void Backpropagate(Vector inputs, Vector errors, Vector outputErrors, double learningRate)
    {
        // Clear output errors first
        outputErrors.Zero();
        
        var errorsSpan = errors.AsReadOnlySpan();
        for (int i = 0; i < _neurons.Length; i++)
        {
            _neurons[i].Backpropagate(inputs, errorsSpan[i], _neuronErrors, learningRate);
            outputErrors.AddInPlace(_neuronErrors);
        }
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _neurons.Length; i++)
            sb.AppendLine($"  Neuron {i + 1}: {_neurons[i]}");

        return sb.ToString();
    }
}