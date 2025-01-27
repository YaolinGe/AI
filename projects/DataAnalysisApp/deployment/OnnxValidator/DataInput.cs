// DataFrame.cs
using System;
using System.Collections.Generic;
using System.Linq;

public class DataInput
{
    public DataInput Diff()
    {
        var diffDf = new DataInput();
        foreach (var (name, values) in _columns)
        {
            var diffValues = values.Skip(1).Select((v, i) => v - values[i]).ToList();
            diffDf.AddColumn(name, diffValues);
        }
        return diffDf;
    }

    public DataInput DropNA()
    {
        var cleanDf = new DataInput();
        var validIndices = Enumerable.Range(0, RowCount)
            .Where(i => _columns.Values.All(col => !double.IsNaN(col[i])))
            .ToList();

        foreach (var (name, values) in _columns)
        {
            cleanDf.AddColumn(name, validIndices.Select(i => values[i]));
        }
        return cleanDf;
    }

    public DataInput Slice(int start, int end)
    {
        var sliced = new DataInput();
        foreach (var (name, values) in _columns)
        {
            sliced.AddColumn(name, values.Skip(start).Take(end - start));
        }
        return sliced;
    }

    public double[][] ToArray() =>
        Enumerable.Range(0, RowCount)
            .Select(i => _columns.Values.Select(col => col[i]).ToArray())
            .ToArray();
}