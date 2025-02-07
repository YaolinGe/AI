// MinMaxScaler.cs
using System.Collections.Generic;
using System.Linq;

namespace OnnxValidator;


public class MinMaxScaler
{
    private readonly double targetMin;
    private readonly double targetMax;

    public MinMaxScaler(double? min = 0, double? max = 1)
    {
        targetMin = min ?? 0;
        targetMax = max ?? 1;
    }

    public double[,] Transform(double[,] data,
                               Dictionary<int, (double min, double max)> customRanges)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(customRanges);

        int rows = data.GetLength(0);
        int columns = data.GetLength(1);
        double[,] transformed = new double[rows, columns];

        for (int col = 0; col < columns; col++)
        {
            double dataMin = customRanges[col].min;
            double dataMax = customRanges[col].max;

            for (int row = 0; row < rows; row++)
            {
                transformed[row, col] = (data[row, col] - dataMin) / (dataMax - dataMin) * (targetMax - targetMin) + targetMin;
            }
        }

        return transformed;
    }
}
