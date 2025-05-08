using System.Text;

namespace BackPropNN;

class NeuralNetworkImpl : NeuralNetwork
{
    private List<Layer> _layers = []; 

    public NeuralNetworkImpl(NeuralNetworkFactory factory, params int[] layerSizes)
    {
        for (int i = 1; i < layerSizes.Length; i++)
            _layers.Add(factory.CreateLayer(layerSizes[i], layerSizes[i - 1]));
    }

    private List<Vector> _activations = new List<Vector>();

    public Vector FeedForward(Vector inputs) 
    { 
        _activations = new List<Vector> { inputs };
        return _layers.Aggregate(inputs, (currentOutputs, layer) => {
            var output = layer.FeedForward(currentOutputs);
            _activations.Add(output);
            return output;
        });
    }

    public void BackPropagate(Vector initialInputs, Vector outputErrors, double learningRate)
    {
            // We assume that FeedForward has been called before this method
            if (_activations.Count == 0)
            {
                // We don't go down this path, but just in case              
                FeedForward(initialInputs);
            }

            var _= Enumerable.Range(0, _layers.Count)
                    .Reverse()
                    .Aggregate(
                        outputErrors,
                        (currentErrors, i) => _layers[i].Backpropagate(_activations[i], currentErrors, learningRate)
                    );   
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
