using System.Text;
using System.Runtime.CompilerServices;
using System.Numerics;

namespace BackPropNN;

/// <summary>
/// Optimized layer implementation using Structure-of-Arrays (SoA) approach for better cache locality and vectorization
/// </summary>
class OptimizedLayer : Layer
{
    private readonly int _neuronCount;
    private readonly int _inputSize;
    
    // Structure-of-Arrays: All data of the same type stored together
    private readonly double[] _allWeights;    // All weights in one contiguous array
    private readonly double[] _allBiases;     // All biases in one contiguous array
    private readonly double[] _outputs;       // Pre-allocated outputs
    private readonly double[] _lastOutputs;   // For backpropagation (cached sigmoid outputs)
    
    private readonly Vector _outputVector;    // Wrapper for outputs array
    
    private static readonly Random _random = new Random();

    public OptimizedLayer(int neuronCount, int inputSize)
    {
        _neuronCount = neuronCount;
        _inputSize = inputSize;
        
        // Allocate all arrays
        _allWeights = new double[neuronCount * inputSize];
        _allBiases = new double[neuronCount];
        _outputs = new double[neuronCount];
        _lastOutputs = new double[neuronCount];
        
        _outputVector = new Vector(_outputs, takeOwnership: true);
        
        // Initialize weights and biases using Xavier initialization
        InitializeWeightsAndBiases();
    }

    private void InitializeWeightsAndBiases()
    {
        double sqrtVariance = Math.Sqrt(1.0 / _inputSize);
        
        // Initialize weights
        for (int i = 0; i < _allWeights.Length; i++)
        {
            _allWeights[i] = _random.NextGaussian(0, sqrtVariance);
        }
        
        // Initialize biases
        for (int i = 0; i < _allBiases.Length; i++)
        {
            _allBiases[i] = _random.NextGaussian(0, sqrtVariance);
        }
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    
    private static double SigmoidDerivative(double output) => output * (1 - output);

    /// <summary>
    /// SIMD-optimized dot product for a single neuron
    /// </summary>
    private static unsafe double SIMDDotProduct(double* weightsPtr, double* inputPtr, int length)
    {
        int vectorSize = Vector<double>.Count;
        int vectorizedLength = length - (length % vectorSize);
        
        var sumVector = Vector<double>.Zero;
        
        // Vectorized dot product
        for (int i = 0; i < vectorizedLength; i += vectorSize)
        {
            var weightVec = Unsafe.Read<Vector<double>>(weightsPtr + i);
            var inputVec = Unsafe.Read<Vector<double>>(inputPtr + i);
            sumVector += weightVec * inputVec;
        }
        
        // Sum the vector elements
        double sum = 0;
        for (int i = 0; i < vectorSize; i++)
        {
            sum += sumVector[i];
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < length; i++)
        {
            sum += weightsPtr[i] * inputPtr[i];
        }
        
        return sum;
    }

    public Vector FeedForward(Vector inputs)
    {
        var inputSpan = inputs.AsReadOnlySpan();
        
        unsafe
        {
            fixed (double* weightsPtr = _allWeights)
            fixed (double* biasesPtr = _allBiases)
            fixed (double* outputsPtr = _outputs)
            fixed (double* lastOutputsPtr = _lastOutputs)
            fixed (double* inputPtr = inputSpan)
            {
                // Process all neurons in batch for better cache locality
                for (int neuron = 0; neuron < _neuronCount; neuron++)
                {
                    double* neuronWeights = weightsPtr + (neuron * _inputSize);
                    
                    // SIMD-optimized dot product
                    double sum = biasesPtr[neuron] + SIMDDotProduct(neuronWeights, inputPtr, _inputSize);
                    
                    double output = Sigmoid(sum);
                    outputsPtr[neuron] = output;
                    lastOutputsPtr[neuron] = output; // Cache for backpropagation
                }
            }
        }
        
        return _outputVector;
    }

    public void Backpropagate(Vector inputs, Vector errors, Vector outputErrors, double learningRate)
    {
        // Clear output errors first
        outputErrors.Zero();
        
        var inputSpan = inputs.AsReadOnlySpan();
        var errorsSpan = errors.AsReadOnlySpan();
        var outputErrorsSpan = outputErrors.AsSpan();
        
        unsafe
        {
            fixed (double* weightsPtr = _allWeights)
            fixed (double* biasesPtr = _allBiases)
            fixed (double* lastOutputsPtr = _lastOutputs)
            fixed (double* inputPtr = inputSpan)
            fixed (double* errorsPtr = errorsSpan)
            fixed (double* outputErrorsPtr = outputErrorsSpan)
            {
                // Process all neurons in batch
                for (int neuron = 0; neuron < _neuronCount; neuron++)
                {
                    double sigmoidDeriv = SigmoidDerivative(lastOutputsPtr[neuron]);
                    double delta = errorsPtr[neuron] * sigmoidDeriv;
                    double* neuronWeights = weightsPtr + (neuron * _inputSize);
                    
                    // SIMD-optimized error accumulation and weight updates
                    int vectorSize = Vector<double>.Count;
                    int vectorizedLength = _inputSize - (_inputSize % vectorSize);
                    var deltaVector = new Vector<double>(delta);
                    var learningRateVector = new Vector<double>(learningRate);
                    var learningRateDeltaVector = learningRateVector * deltaVector;
                    
                    // Vectorized operations
                    for (int i = 0; i < vectorizedLength; i += vectorSize)
                    {
                        // Load current values
                        var weightVec = Unsafe.Read<Vector<double>>(neuronWeights + i);
                        var inputVec = Unsafe.Read<Vector<double>>(inputPtr + i);
                        var outputErrorVec = Unsafe.Read<Vector<double>>(outputErrorsPtr + i);
                        
                        // Accumulate output errors: outputErrors[i] += weights[neuron][i] * delta
                        var errorContribution = weightVec * deltaVector;
                        Unsafe.Write(outputErrorsPtr + i, outputErrorVec + errorContribution);
                        
                        // Update weights: weights[neuron][i] += learningRate * delta * inputs[i]
                        var weightUpdate = learningRateDeltaVector * inputVec;
                        Unsafe.Write(neuronWeights + i, weightVec + weightUpdate);
                    }
                    
                    // Handle remaining elements
                    for (int i = vectorizedLength; i < _inputSize; i++)
                    {
                        // Accumulate output errors: outputErrors[i] += weights[neuron][i] * delta
                        outputErrorsPtr[i] += neuronWeights[i] * delta;
                        
                        // Update weights: weights[neuron][i] += learningRate * delta * inputs[i]
                        neuronWeights[i] += learningRate * delta * inputPtr[i];
                    }
                    
                    // Update bias: bias[neuron] += learningRate * delta
                    biasesPtr[neuron] += learningRate * delta;
                }
            }
        }
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (int neuron = 0; neuron < _neuronCount; neuron++)
        {
            sb.Append($"  Neuron {neuron + 1}: Weights=[");
            
            for (int i = 0; i < _inputSize; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append($"W{i + 1}: {_allWeights[neuron * _inputSize + i]:F6}");
            }
            
            sb.AppendLine($"], Bias: {_allBiases[neuron]:F6}");
        }
        return sb.ToString();
    }
} 