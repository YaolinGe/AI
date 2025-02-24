using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Sandvik.Coromant.CoroPlus.Tooling.SilentTools.BlazorApp.Pages.Playground.DevelopmentModules.MicrosoftPoC;

public class InferenceEngine : IDisposable
{
    private readonly InferenceSession _lstmSession;
    private readonly InferenceSession _classicalSession;
    private readonly double _threshold = 8.156088188115973e-05;

    public InferenceEngine(string modelBasePath, bool useGpu = false)
    {
        string lstmModelPath = Path.Combine(modelBasePath, "LSTM_AD.onnx");
        string classicalModelPath = Path.Combine(modelBasePath, "GNB.onnx");

        if (!File.Exists(lstmModelPath))
            throw new FileNotFoundException($"LSTM model not found at {lstmModelPath}");
        if (!File.Exists(classicalModelPath))
            throw new FileNotFoundException($"Classical model not found at {classicalModelPath}");

        var sessionOptions = new SessionOptions();
        if (useGpu)
        {
            sessionOptions.AppendExecutionProvider_CUDA();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableProfiling = true;
        }

        _lstmSession = new InferenceSession(lstmModelPath, sessionOptions);
        _classicalSession = new InferenceSession(classicalModelPath, sessionOptions);
    }

    public (bool[], double[,]) RunInferenceUsingLSTM(double[,,] input)
    {
        int n = input.GetLength(0);
        int m = input.GetLength(1);
        int p = input.GetLength(2);
        var inputTensor = new DenseTensor<float>(new[] { n, m, p });

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int k = 0; k < p; k++)
                    inputTensor[i, j, k] = (float)input[i, j, k];

        string inputName = _lstmSession.InputMetadata.Keys.First();
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
        using var results = _lstmSession.Run(inputs);
        var outputTensor = results.First().Value as DenseTensor<float>;

        double[,] errorSum = new double[1, p];
        bool[] result = new bool[n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    double error = Math.Pow(outputTensor[i, j, k] - input[i, j, k], 2);
                    errorSum[0, k] += error;
                    if (error > _threshold)
                        result[i] = true;
                }
            }
        }

        return (result, errorSum);
    }

    public bool[] RunInferenceUsingClassicalModel(double[,] inputData)
    {
        int n = inputData.GetLength(0);
        int m = inputData.GetLength(1);
        var inputTensor = new DenseTensor<float>(new[] { n, m });

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                inputTensor[i, j] = (float)inputData[i, j];

        string inputName = _classicalSession.InputMetadata.Keys.First();
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
        using var results = _classicalSession.Run(inputs);
        var resultValue = results.First().Value as DenseTensor<long>;

        return resultValue.ToArray().Select(x => x != 0).ToArray();
    }

    public void Dispose()
    {
        _lstmSession?.Dispose();
        _classicalSession?.Dispose();
    }
}