using System.Text;

namespace BackPropNN;

class LayerImpl : Layer
{
    private Neuron[] _neurons;

    private Vector _output;

    public LayerImpl(int neuronCount, int inputsPerNeuron, Func<int, Neuron> funcFactory)
    {
        _neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            _neurons[i] = funcFactory(inputsPerNeuron); 
        _output = new Vector(_neurons.Length);
    }

    public Vector FeedForward(Vector inputs)
    {
        for (int i = 0; i < _neurons.Length; i++)
        {
            _output[i] = _neurons[i].FeedForward(inputs);
        }
        return _output;
    }

    public void Backpropagate(Vector inputs, Vector errors, Vector outputErrors, double learningRate)
    {
        // Clear output errors first
        outputErrors.Zero();

        // Create a temporary buffer for each neuron's errors
        Vector neuronErrors = new Vector(inputs.Length);
        
        for (int i = 0; i < _neurons.Length; i++)
        {
            _neurons[i].Backpropagate(inputs, errors[i], neuronErrors, learningRate);
            outputErrors.AddInPlace(neuronErrors);
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