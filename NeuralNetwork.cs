using System.Text;

namespace BackPropNN;

class NeuralNetworkImpl : NeuralNetwork
{
    private List<Layer> _layers = new List<Layer>();

    public NeuralNetworkImpl(NeuralNetworkFactory factory, params int[] layerSizes)
    {
        // Create layers based on the provided sizes
        for (int i = 1; i < layerSizes.Length; i++)
        {
            _layers.Add(factory.CreateLayer(layerSizes[i], layerSizes[i - 1]));
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

    public void BackPropagate(Vector initialInputs, Vector outputErrors, double learningRate)
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
