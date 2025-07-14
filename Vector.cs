using System.Numerics;
using System.Runtime.CompilerServices;

namespace BackPropNN;

class Vector
{
    private double[] _elements;

    public Vector(int size)
    {
        _elements = new double[size];
    }

    public Vector(double[] elements, bool takeOwnership)
    {
        if (takeOwnership)
        {
            _elements = elements;
        }
        else
        {
            _elements = new double[elements.Length];
            Array.Copy(elements, _elements, elements.Length);
        }
    }

    public Vector(Vector vector, bool takeOwnership)
    {
        if (takeOwnership)
        {
            _elements = vector._elements;
        }
        else
        {
            _elements = new double[vector._elements.Length];
            Array.Copy(vector._elements, _elements, vector._elements.Length);
        }
    }    

    public double this[int index]
    {
        get { return _elements[index]; }
        set { _elements[index] = value; }
    }

    public int Length => _elements.Length;

    public Span<double> AsSpan() => _elements;
    
    public ReadOnlySpan<double> AsReadOnlySpan() => _elements;

    public void CopyFrom(Vector source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Vectors must be the same size for copy");
        
        unsafe
        {
            var sourceSpan = source.AsReadOnlySpan();
            var destSpan = AsSpan();
            
            fixed (double* srcPtr = sourceSpan)
            fixed (double* destPtr = destSpan)
            {
                Buffer.MemoryCopy(srcPtr, destPtr, Length * sizeof(double), Length * sizeof(double));
            }
        }
    }

    public void CopyFrom(double[] source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Array must be the same size for copy");
        
        unsafe
        {
            var sourceSpan = source.AsSpan();
            var destSpan = AsSpan();
            
            fixed (double* srcPtr = sourceSpan)
            fixed (double* destPtr = destSpan)
            {
                Buffer.MemoryCopy(srcPtr, destPtr, Length * sizeof(double), Length * sizeof(double));
            }
        }
    }

    public void CopyFrom(ReadOnlySpan<double> source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Span must be the same size for copy");
        
        unsafe
        {
            var destSpan = AsSpan();
            
            fixed (double* srcPtr = source)
            fixed (double* destPtr = destSpan)
            {
                Buffer.MemoryCopy(srcPtr, destPtr, Length * sizeof(double), Length * sizeof(double));
            }
        }
    }

    public void Zero()
    {
        unsafe
        {
            var span = AsSpan();
            fixed (double* ptr = span)
            {
                Unsafe.InitBlock(ptr, 0, (uint)(Length * sizeof(double)));
            }
        }
    }

    public void AddInPlace(Vector other)
    {
        if (other.Length != Length)
            throw new ArgumentException("Vectors must be the same length for addition");
        
        AddInPlace(other.AsReadOnlySpan());
    }

    public void AddInPlace(ReadOnlySpan<double> other)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for addition");
        
        var thisSpan = AsSpan();
        
        unsafe
        {
            fixed (double* thisPtr = thisSpan)
            fixed (double* otherPtr = other)
            {
                int vectorSize = Vector<double>.Count;
                int vectorizedLength = thisSpan.Length - (thisSpan.Length % vectorSize);
                
                // Vectorized addition using unsafe pointers
                for (int i = 0; i < vectorizedLength; i += vectorSize)
                {
                    var v1 = Unsafe.Read<Vector<double>>(thisPtr + i);
                    var v2 = Unsafe.Read<Vector<double>>(otherPtr + i);
                    Unsafe.Write(thisPtr + i, v1 + v2);
                }
                
                // Handle remaining elements
                for (int i = vectorizedLength; i < thisSpan.Length; i++)
                {
                    thisPtr[i] += otherPtr[i];
                }
            }
        }
    }

    public static double DotProduct(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Spans must be the same size for dot product");

        unsafe
        {
            fixed (double* v1Ptr = v1)
            fixed (double* v2Ptr = v2)
            {
                int vectorSize = Vector<double>.Count;
                int vectorizedLength = v1.Length - (v1.Length % vectorSize);
                
                var sumVector = Vector<double>.Zero;
                
                // Vectorized dot product using unsafe pointers
                for (int i = 0; i < vectorizedLength; i += vectorSize)
                {
                    var vec1 = Unsafe.Read<Vector<double>>(v1Ptr + i);
                    var vec2 = Unsafe.Read<Vector<double>>(v2Ptr + i);
                    sumVector += vec1 * vec2;
                }
                
                // Sum the vector elements
                double sum = 0;
                for (int i = 0; i < vectorSize; i++)
                {
                    sum += sumVector[i];
                }
                
                // Handle remaining elements
                for (int i = vectorizedLength; i < v1.Length; i++)
                {
                    sum += v1Ptr[i] * v2Ptr[i];
                }

                return sum;
            }
        }
    }

    public void MultiplyElementWise(ReadOnlySpan<double> other, double scalar)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for element-wise multiplication");
        
        var thisSpan = AsSpan();
        
        unsafe
        {
            fixed (double* thisPtr = thisSpan)
            fixed (double* otherPtr = other)
            {
                int vectorSize = Vector<double>.Count;
                int vectorizedLength = thisSpan.Length - (thisSpan.Length % vectorSize);
                var scalarVector = new Vector<double>(scalar);
                
                // Vectorized multiplication using unsafe pointers
                for (int i = 0; i < vectorizedLength; i += vectorSize)
                {
                    var otherVec = Unsafe.Read<Vector<double>>(otherPtr + i);
                    var result = otherVec * scalarVector;
                    Unsafe.Write(thisPtr + i, result);
                }
                
                // Handle remaining elements
                for (int i = vectorizedLength; i < thisSpan.Length; i++)
                {
                    thisPtr[i] = otherPtr[i] * scalar;
                }
            }
        }
    }

    public void AddScaledElementWise(ReadOnlySpan<double> other, double scalar)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for scaled addition");
        
        var thisSpan = AsSpan();
        
        unsafe
        {
            fixed (double* thisPtr = thisSpan)
            fixed (double* otherPtr = other)
            {
                int vectorSize = Vector<double>.Count;
                int vectorizedLength = thisSpan.Length - (thisSpan.Length % vectorSize);
                var scalarVector = new Vector<double>(scalar);
                
                // Vectorized scaled addition using unsafe pointers
                for (int i = 0; i < vectorizedLength; i += vectorSize)
                {
                    var thisVec = Unsafe.Read<Vector<double>>(thisPtr + i);
                    var otherVec = Unsafe.Read<Vector<double>>(otherPtr + i);
                    var result = thisVec + (otherVec * scalarVector);
                    Unsafe.Write(thisPtr + i, result);
                }
                
                // Handle remaining elements
                for (int i = vectorizedLength; i < thisSpan.Length; i++)
                {
                    thisPtr[i] += otherPtr[i] * scalar;
                }
            }
        }
    }

    // Create a new vector from a span
    public static Vector FromSpan(ReadOnlySpan<double> span)
    {
        var vector = new Vector(span.Length);
        span.CopyTo(vector.AsSpan());
        return vector;
    }
    
    public override string ToString()
    {
        return string.Join(", ", _elements.Select((w, i) => $"W{i + 1}: {w:F6}"));
    }
}
