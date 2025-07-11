using System.Text;

namespace BackPropNN;

class NeuralNetworkImpl : NeuralNetwork
{
    private List<Layer> _layers = []; 
    private List<Vector> _activations = new List<Vector>();

    public NeuralNetworkImpl(NeuralNetworkFactory factory, params int[] layerSizes)
    {
        for (int i = 1; i < layerSizes.Length; i++)
            _layers.Add(factory.CreateLayer(layerSizes[i], layerSizes[i - 1]));
    }

    public Vector FeedForward(Vector inputs) 
    { 
        // Clear and reuse activation list
        _activations.Clear();
        _activations.Add(inputs);
        
        Vector currentOutputs = inputs;
        for (int i = 0; i < _layers.Count; i++)
        {
            var output = _layers[i].FeedForward(currentOutputs);
            _activations.Add(output);
            currentOutputs = output;
        }
        
        return currentOutputs;
    }

    public void BackPropagate(Vector initialInputs, Vector outputErrors, double learningRate)
    {
        // We assume that FeedForward has been called before this method
        if (_activations.Count == 0)
        {
            // We don't go down this path, but just in case              
            FeedForward(initialInputs);
        }

        Vector currentErrors = outputErrors;
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            Vector layerErrors = new Vector(_activations[i].Length); // Buffer for layer errors, sized to this layer's input
            _layers[i].Backpropagate(_activations[i], currentErrors, layerErrors, learningRate);
            currentErrors = layerErrors;
        }
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < _layers.Count; i++)
        {
            sb.AppendLine($"Layer {i + 1}:");
            sb.AppendLine(_layers[i].ToString());
        }
        return sb.ToString();
    }
}
