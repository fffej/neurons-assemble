using System.Text;

namespace BackPropNN;

class Layer
{
    private Neuron[] _neurons;

    public Layer(int neuronCount, int inputsPerNeuron, Func<int, Neuron> funcFactory)
    {
        _neurons = new MultiInputNeuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
        {
            _neurons[i] = funcFactory(inputsPerNeuron); 
        }
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
        // For hidden layers, we need to calculate errors for the previous layer
        Vector previousLayerErrors = new Vector(inputs.Length);

        // Update each neuron and accumulate errors for previous layer
        for (int i = 0; i < _neurons.Length; i++)
        {
            // Get error contributions to previous layer from this neuron
            var neuronErrors = _neurons[i].Backpropagate(inputs, errors[i], learningRate);

            // Accumulate errors for previous layer
            for (int j = 0; j < previousLayerErrors.Length; j++)
            {
                previousLayerErrors[j] += neuronErrors[j];
            }
        }

        return previousLayerErrors;
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < _neurons.Length; i++)
        {
            sb.AppendLine($"  Neuron {i + 1}: {_neurons[i]}");
        }
        return sb.ToString();
    }
}