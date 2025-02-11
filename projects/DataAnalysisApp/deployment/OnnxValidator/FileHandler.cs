using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxValidator
{
    public class FileHandler
    {
        public double[,] LoadData(string path)
        {
            string[] csvLines = System.IO.File.ReadAllLines(path);
            int rows = csvLines.Length - 1;
            int cols = csvLines[0].Split(',').Length;
            double[,] data = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                double[] values = csvLines[i + 1].Split(',').Select(double.Parse).ToArray();
                for (int j = 0; j < cols; j++)
                {
                    data[i, j] = values[j];
                }
            }
            System.Diagnostics.Debug.WriteLine($"Loaded data from {path} with {rows} rows and {cols} columns.");
            return data;
        }

        public Dictionary<int, (double min, double max)> LoadMinMaxValues(string path)
        {
            string[] lines = System.IO.File.ReadAllLines(path);
            Dictionary<int, (double min, double max)> customRanges = new();
            for (int i = 0; i < lines.Length - 1; i++)
            {
                string[] values = lines[i + 1].Split(',');
                customRanges[i] = (double.Parse(values[0]), double.Parse(values[1]));
            }
            System.Diagnostics.Debug.WriteLine($"Loaded min max values from {path}.");
            foreach (var kvp in customRanges)
            {
                System.Diagnostics.Debug.WriteLine($"Column {kvp.Key}: Min = {kvp.Value.min}, Max = {kvp.Value.max}");
            }
            return customRanges;
        }

        public void SaveData(string path, double[,] data)
        {
            StringBuilder sb = new();
            int rows = data.GetLength(0);
            int cols = data.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    sb.Append(data[i, j]);
                    if (j < cols - 1)
                    {
                        sb.Append(",");
                    }
                }   
                sb.AppendLine();
            }
            System.IO.File.WriteAllText(path, sb.ToString());
        }
    }
}
