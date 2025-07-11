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
        
        var thisSpan = AsSpan();
        var otherSpan = other.AsReadOnlySpan();
        
        for (int i = 0; i < thisSpan.Length; i++)
        {
            thisSpan[i] += otherSpan[i];
        }
    }

    public void AddInPlace(ReadOnlySpan<double> other)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for addition");
        
        var thisSpan = AsSpan();
        
        for (int i = 0; i < thisSpan.Length; i++)
        {
            thisSpan[i] += other[i];
        }
    }

    public static double DotProduct(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Spans must be the same size for dot product");

        double sum = 0;
        for (int i = 0; i < v1.Length; i++)
            sum += v1[i] * v2[i];

        return sum;
    }

    public void MultiplyElementWise(ReadOnlySpan<double> other, double scalar)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for element-wise multiplication");
        
        var thisSpan = AsSpan();
        
        for (int i = 0; i < thisSpan.Length; i++)
        {
            thisSpan[i] = other[i] * scalar;
        }
    }

    public void AddScaledElementWise(ReadOnlySpan<double> other, double scalar)
    {
        if (other.Length != Length)
            throw new ArgumentException("Span must be the same length for scaled addition");
        
        var thisSpan = AsSpan();
        
        for (int i = 0; i < thisSpan.Length; i++)
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
