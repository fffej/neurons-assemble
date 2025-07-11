namespace BackPropNN;

class NeuralNetworkFactoryImpl : NeuralNetworkFactory
{
    public Neuron CreateNeuron(int inputSize) => new MultiInputNeuron(inputSize);
    public Layer CreateLayer(int neuronCount, int inputsPerNeuron) => new LayerImpl(neuronCount, inputsPerNeuron, CreateNeuron);
} 