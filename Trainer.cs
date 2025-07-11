using System.Diagnostics;

namespace BackPropNN;

class Trainer
{
    private NeuralNetwork _network;
    private double[][] _trainingInputs;
    private double[][] _expectedOutputs;
    private double _learningRate;
    private int _epochs;
    
    // Pre-allocated vectors to avoid repeated allocations
    private Vector _inputVector;
    private Vector _outputVector;
    private Vector _errorVector;

    public Trainer(NeuralNetwork network, double[][] trainingInputs, double[][] expectedOutputs, double learningRate, int epochs)
    {
        _network = network;
        _trainingInputs = trainingInputs;
        _expectedOutputs = expectedOutputs;
        _learningRate = learningRate;
        _epochs = epochs;
        
        // Pre-allocate vectors (assume they are the same size)
        int maxInputSize = trainingInputs[0].Length;
        int maxOutputSize = expectedOutputs[0].Length;
        
        _inputVector = new Vector(maxInputSize);
        _outputVector = new Vector(maxOutputSize);
        _errorVector = new Vector(maxOutputSize);
    }

    public void Train()
    {
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            TrainSingleIteration();
        }
    }

    public void TrainSingleIteration()
    {
        double totalError = 0;

        // Train on each input-output pair
        for (int i = 0; i < _trainingInputs.Length; i++)
        {
            _inputVector.CopyFrom(_trainingInputs[i]);
            var outputs = _network.FeedForward(_inputVector);
            
            _outputVector.CopyFrom(outputs);
            
            _errorVector.Zero();
            for (int j = 0; j < outputs.Length; j++)
            {                    
                _errorVector[j] = _expectedOutputs[i][j] - _outputVector[j];
                totalError += Math.Abs(_errorVector[j]);
            }

            _network.BackPropagate(_inputVector, _errorVector, _learningRate);
        }
    }
}
