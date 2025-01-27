// MinMaxScaler.cs
using System.Collections.Generic;
using System.Linq;

public class MinMaxScaler
{
    private readonly Dictionary<string, (double min, double max)> _ranges = new();

    public void Fit(DataInput df)
    {
        foreach (var col in df.Columns)
        {
            var values = df.GetColumn(col);
            _ranges[col] = (values.Min(), values.Max());
        }
    }

    public DataInput Transform(DataInput df)
    {
        var transformed = new DataInput();
        foreach (var col in df.Columns)
        {
            var (min, max) = _ranges[col];
            transformed.AddColumn(col, df.GetColumn(col).Select(v => (v - min) / (max - min)));
        }
        return transformed;
    }
    
    public DataInput FitTransform(DataInput df)
    {
        Fit(df);
        return Transform(df);
    }
    
}