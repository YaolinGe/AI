// MinMaxScaler.cs
using System.Collections.Generic;
using System.Linq;

namespace OnnxValidator;


public class MinMaxScaler
{
    public Dictionary<int, (double min, double max)> Fit(double[,] data)
    {
        Dictionary<int, (double min, double max)> ranges = [];
        int columns = data.GetLength(1);
        for (int col = 0; col < columns; col++)
        {
            var values = Enumerable.Range(0, data.GetLength(0)).Select(row => data[row, col]);
            ranges[col] = (values.Min(), values.Max());
        }
        return ranges; 
    }

    public double[,] Transform(double[,] data, Dictionary<int, (double min, double max)>? customRanges = null)
    {
        int rows = data.GetLength(0);
        int columns = data.GetLength(1);
        double[,] transformed = new double[rows, columns];

        Dictionary<int, (double min, double max)> dataRanges = Fit(data);

        for (int col = 0; col < columns; col++)
        {
            for (int row = 0; row < rows; row++)
            {
                transformed[row, col] = (data[row, col] - dataRanges[col].min) / (dataRanges[col].max - dataRanges[col].min) * (customRanges != null ? customRanges[col].max - customRanges[col].min : 1) + (customRanges != null ? customRanges[col].min : 0);
            }
        }

        return transformed;
    }
}
