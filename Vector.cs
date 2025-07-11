using System.Numerics;

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
        
        var sourceSpan = source.AsReadOnlySpan();
        var destSpan = AsSpan();
        sourceSpan.CopyTo(destSpan);
    }

    public void CopyFrom(double[] source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Array must be the same size for copy");
        
        var sourceSpan = source.AsSpan();
        var destSpan = AsSpan();
        sourceSpan.CopyTo(destSpan);
    }

    public void CopyFrom(ReadOnlySpan<double> source)
    {
        if (source.Length != Length)
            throw new ArgumentException("Span must be the same size for copy");
        
        var destSpan = AsSpan();
        source.CopyTo(destSpan);
    }

    public void Zero()
    {
        var span = AsSpan();
        span.Clear();
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
        int vectorSize = Vector<double>.Count;
        int vectorizedLength = thisSpan.Length - (thisSpan.Length % vectorSize);
        
        // Vectorized addition
        for (int i = 0; i < vectorizedLength; i += vectorSize)
        {
            var v1 = new Vector<double>(thisSpan.Slice(i, vectorSize));
            var v2 = new Vector<double>(other.Slice(i, vectorSize));
            var result = v1 + v2;
            result.CopyTo(thisSpan.Slice(i, vectorSize));
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < thisSpan.Length; i++)
        {
            thisSpan[i] += other[i];
        }
    }

    public static double DotProduct(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Spans must be the same size for dot product");

        int vectorSize = Vector<double>.Count;
        int vectorizedLength = v1.Length - (v1.Length % vectorSize);
        
        var sumVector = Vector<double>.Zero;
        
        // Vectorized dot product
        for (int i = 0; i < vectorizedLength; i += vectorSize)
        {
            var vec1 = new Vector<double>(v1.Slice(i, vectorSize));
            var vec2 = new Vector<double>(v2.Slice(i, vectorSize));
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
            sum += v1[i] * v2[i];
        }

        return sum;
    }

    public void MultiplyElementWise(ReadOnlySpan<double> other, double scalar)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for element-wise multiplication");
        
        var thisSpan = AsSpan();
        int vectorSize = Vector<double>.Count;
        int vectorizedLength = thisSpan.Length - (thisSpan.Length % vectorSize);
        var scalarVector = new Vector<double>(scalar);
        
        // Vectorized multiplication
        for (int i = 0; i < vectorizedLength; i += vectorSize)
        {
            var otherVec = new Vector<double>(other.Slice(i, vectorSize));
            var result = otherVec * scalarVector;
            result.CopyTo(thisSpan.Slice(i, vectorSize));
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < thisSpan.Length; i++)
        {
            thisSpan[i] = other[i] * scalar;
        }
    }

    public void AddScaledElementWise(ReadOnlySpan<double> other, double scalar)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for scaled addition");
        
        var thisSpan = AsSpan();
        int vectorSize = Vector<double>.Count;
        int vectorizedLength = thisSpan.Length - (thisSpan.Length % vectorSize);
        var scalarVector = new Vector<double>(scalar);
        
        // Vectorized scaled addition
        for (int i = 0; i < vectorizedLength; i += vectorSize)
        {
            var thisVec = new Vector<double>(thisSpan.Slice(i, vectorSize));
            var otherVec = new Vector<double>(other.Slice(i, vectorSize));
            var result = thisVec + (otherVec * scalarVector);
            result.CopyTo(thisSpan.Slice(i, vectorSize));
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < thisSpan.Length; i++)
        {
            thisSpan[i] += other[i] * scalar;
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
