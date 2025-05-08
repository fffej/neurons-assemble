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

    public Vector FeedForward(Vector inputs) => 
        new Vector(_neurons.Select(n => n.FeedForward(inputs)).ToArray());

    public Vector Backpropagate(Vector inputs, Vector errors, double learningRate) =>
        _neurons.Select((neuron, i) => neuron.Backpropagate(inputs, errors[i], learningRate))
                  .Aggregate(
                      new Vector(inputs.Length), // Start with a zero vector of the right size
                      (accumulatedErrors, neuronErrors) => accumulatedErrors + neuronErrors
                  );

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _neurons.Length; i++)
            sb.AppendLine($"  Neuron {i + 1}: {_neurons[i]}");

        return sb.ToString();
    }
}