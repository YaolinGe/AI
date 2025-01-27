// DataProcessor.cs
using System;
using System.Collections.Generic;
using System.Linq;

public class DataProcessor
{
    private readonly int[] _classicalLags = { 5, 10 };
    private readonly int[] _windows = { 30 };

    public double[,] Process(double[,] data)
    {
        // Step 1: Select and scale raw columns
        var scaled = new MinMaxScaler().FitTransform(data);

        // Step 2: Compute first difference
        var differenced = scaled.Diff();

        // Step 3: Clean data
        var cleaned = differenced.DropNA();

        // Step 4: Split data
        // var (train, val, test) = SplitData(cleaned);

        // Step 5: Create features
        return new ProcessedData(
            CreateClassicalFeatures(train, val, test),
            CreateLSTMFeatures(cleaned, train.RowCount, val.RowCount, test.RowCount)
        );
    }
    
    static double[,] ComputeFirstDifference(double[,] input)
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);

        if (rows < 2)
            throw new ArgumentException("Input matrix must have at least two rows.");

        double[,] diff = new double[rows - 1, cols];

        for (int i = 1; i < rows; i++)  // Start from the second row
        {
            for (int j = 0; j < cols; j++)
            {
                diff[i - 1, j] = input[i, j] - input[i - 1, j];
            }
        }

        return diff;
    }

    // private (DataFrame train, DataFrame val, DataFrame test) SplitData(DataFrame df)
    // {
    //     var trainEnd = (int)(df.RowCount * _splitRatios[0]);
    //     var valEnd = trainEnd + (int)(df.RowCount * _splitRatios[1]);
    //     
    //     return (
    //         df.Slice(0, trainEnd),
    //         df.Slice(trainEnd, valEnd),
    //         df.Slice(valEnd, df.RowCount)
    //     );
    // }
    
    private DataInput AddFeatures(DataInput df)
    {
        var enhanced = df.SelectColumns(_rawColumns);
        
        // Add lag features
        foreach (var lag in _classicalLags)
        {
            foreach (var col in _rawColumns)
            {
                var lagged = df.GetColumn(col)
                    .Select((v, i) => i >= lag ? df.GetColumn(col)[i - lag] : double.NaN)
                    .ToList();
                enhanced.AddColumn($"{col}_lag{lag}", lagged);
            }
        }

        // Add moving averages
        foreach (var window in _windows)
        {
            foreach (var col in _rawColumns)
            {
                var ma = new List<double>();
                for (int i = 0; i < df.RowCount; i++)
                {
                    var start = Math.Max(0, i - window + 1);
                    ma.Add(df.GetColumn(col).Skip(start).Take(i - start + 1).Average());
                }
                enhanced.AddColumn($"{col}_ma{window}", ma);
            }
        }

        return enhanced.DropNA();
    }
}
