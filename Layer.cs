using System.Text;

namespace BackPropNN;

class LayerImpl : Layer
{
    private Neuron[] _neurons;

    public LayerImpl(int neuronCount, int inputsPerNeuron, Func<int, Neuron> funcFactory)
    {
        _neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            _neurons[i] = funcFactory(inputsPerNeuron); 
    }

    public Vector FeedForward(Vector inputs)
    {
        Vector outputs = new Vector(_neurons.Length);
        for (int i = 0; i < _neurons.Length; i++)
        {
            outputs[i] = _neurons[i].FeedForward(inputs);
        }
        return outputs;
    }

    public Vector Backpropagate(Vector inputs, Vector errors, double learningRate)
    {
        Vector accumulatedErrors = new Vector(inputs.Length);
        
        for (int i = 0; i < _neurons.Length; i++)
        {
            Vector neuronErrors = _neurons[i].Backpropagate(inputs, errors[i], learningRate);
            accumulatedErrors.AddInPlace(neuronErrors);
        }
        
        return accumulatedErrors;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _neurons.Length; i++)
            sb.AppendLine($"  Neuron {i + 1}: {_neurons[i]}");

        return sb.ToString();
    }
}