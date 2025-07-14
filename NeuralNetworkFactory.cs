namespace BackPropNN;

class NeuralNetworkFactoryImpl : NeuralNetworkFactory
{
    public Neuron CreateNeuron(int inputSize) => new MultiInputNeuron(inputSize);
    public Layer CreateLayer(int neuronCount, int inputsPerNeuron) => new LayerImpl(neuronCount, inputsPerNeuron, CreateNeuron);
}

/// <summary>
/// Factory for creating optimized neural network components using structure-of-arrays approach
/// </summary>
class OptimizedNeuralNetworkFactory : NeuralNetworkFactory
{
    public Neuron CreateNeuron(int inputSize) => new MultiInputNeuron(inputSize);
    public Layer CreateLayer(int neuronCount, int inputsPerNeuron) => new OptimizedLayer(neuronCount, inputsPerNeuron);
} 