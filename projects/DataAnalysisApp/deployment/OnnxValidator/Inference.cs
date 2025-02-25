// InferenceEngine.cs
using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


namespace OnnxValidator;


public class InferenceEngine : IDisposable
{
    private InferenceSession inferenceSession;

    public InferenceEngine(bool useGpu = false)
    {
        if (useGpu)
        {

            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            inferenceSession = new InferenceSession(@"C:\MR\CoroPlus.Tooling.SilentTools.BlazorApp\ServerAppRunner\wwwroot\microsoftpoc\LSTM_AD.onnx", sessionOptions);
        }
        else
        {
            inferenceSession = new InferenceSession(@"C:\MR\CoroPlus.Tooling.SilentTools.BlazorApp\ServerAppRunner\wwwroot\microsoftpoc\LSTM_AD.onnx");
        }
        
    }

    public bool[] RunInferenceUsingLSTM(double[,,] input)
    {
        double threshold = 8.156088188115973e-05;  // number from python test data
        string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\model";
        string modelPath = Path.Combine(rootFolder, "LSTM_AD.onnx");
        using InferenceSession session = new(modelPath);
        // Create a DenseTensor<float> from the float[,,] array.
        // ONNX Runtime expects the input shape as an int[]; here it's {n, 30, 3}.
        int n = input.GetLength(0);
        int m = input.GetLength(1);
        int p = input.GetLength(2);
        var inputTensor = new DenseTensor<float>(new[] { n, m, p });
        // Copy data from float[,,] to DenseTensor<float>
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    inputTensor[i, j, k] = (float)input[i, j, k];
                }
            }
        }
        // Get input name from the session (assumes the model has one input)
        string inputName = session.InputMetadata.Keys.First();
        // Create NamedOnnxValue for the input
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
        var outputTensor = results.First().Value as DenseTensor<float>;
        // Convert the output DenseTensor<float> to a float[,,] array
        float[,,] output = new float[n, m, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    output[i, j, k] = outputTensor[i, j, k];
                }
            }
        }

        double[,,] error = new double[n, m, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    error[i, j, k] = (output[i, j, k] - input[i, j, k]) * (output[i, j, k] - input[i, j, k]);
                }
            }
        }

        bool[] result = new bool[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = false;
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    result[i] = error[i, j, k] > threshold ? true : false;
                }
            }
        }

        return result;
    }

    public bool[] RunInferenceUsingClassicalModel(double[,] inputData)
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
        var resultValue = results.First().Value as DenseTensor<long>;
        long[] outputTensor = resultValue.ToArray();

        // Convert to boolean array to mark if the anomaly exists or not
        bool[] anomalyExists = new bool[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            anomalyExists[i] = outputTensor[i] != 0;
        }

        return anomalyExists;
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

    public void Dispose()
    {
        inferenceSession?.Dispose();
    }
}


