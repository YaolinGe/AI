// InferenceEngine.cs
using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


namespace OnnxValidator;


public class InferenceEngine
{
    public void RunInference(double[,] inputData)
    {
        string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\model";
        string modelPath = Path.Combine(rootFolder, "GNB.onnx"); 
        using InferenceSession session = new(modelPath);

        float[,] floatData = ConvertDoubleArrayToFloat(inputData);

        // Create a DenseTensor<float> from the float[,] array.
        // ONNX Runtime expects the input shape as an int[]; here it's {n, 20}.
        int n = floatData.GetLength(0);
        int m = floatData.GetLength(1);
        var inputTensor = new DenseTensor<float>(new[] { n, m });

        // Copy data from float[,] to DenseTensor<float>
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                inputTensor[i, j] = floatData[i, j];
            }
        }

        // Get input name from the session (assumes the model has one input)
        string inputName = session.InputMetadata.Keys.First();

        // Create NamedOnnxValue for the input
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
        //var outputTensor = results.First().Value as IEnumerable<float>;
        
        var resultValue = results.First().Value as DenseTensor<long>;
        long[] outputTensor = resultValue.ToArray(); 

    }

    private float[,] ConvertDoubleArrayToFloat(double[,] inputData)
    {
        int n = inputData.GetLength(0);
        int m = inputData.GetLength(1);
        float[,] floatData = new float[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                floatData[i, j] = (float)inputData[i, j];
            }
        }
        return floatData;
    }
}