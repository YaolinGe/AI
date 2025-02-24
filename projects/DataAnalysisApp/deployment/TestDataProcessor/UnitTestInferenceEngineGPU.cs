using Xunit;
using Xunit.Abstractions;
using System;
using System.Diagnostics;
using Sandvik.Coromant.CoroPlus.Tooling.SilentTools.BlazorApp.Pages.Playground.DevelopmentModules.MicrosoftPoC;


public class InferenceEngineTests : IDisposable
{
    private const string ModelBasePath = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\model";
    private readonly ITestOutputHelper _output;
    private readonly InferenceEngine _engine;

    public InferenceEngineTests(ITestOutputHelper output)
    {
        _output = output;
        _engine = new InferenceEngine(ModelBasePath, useGpu: true);
    }

    [Fact]
    public void RunInferenceUsingLSTM_ShouldProcessBatchCorrectly()
    {
        // Arrange
        var input = CreateTestData(batchSize: 100, timeSteps: 30, features: 3);

        // Act
        var sw = Stopwatch.StartNew();
        var (anomalies, errorSum) = _engine.RunInferenceUsingLSTM(input);
        sw.Stop();

        // Assert
        Assert.NotNull(anomalies);
        Assert.Equal(100, anomalies.Length);
        _output.WriteLine($"LSTM Inference Time: {sw.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void RunInferenceUsingClassicalModel_ShouldProcessDataCorrectly()
    {
        // Arrange
        var input = CreateTestData(samples: 100, features: 10);

        // Act
        var sw = Stopwatch.StartNew();
        var predictions = _engine.RunInferenceUsingClassicalModel(input);
        sw.Stop();

        // Assert
        Assert.NotNull(predictions);
        Assert.Equal(100, predictions.Length);
        _output.WriteLine($"Classical Model Inference Time: {sw.ElapsedMilliseconds}ms");
    }

    [Theory]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    public void RunInferenceUsingLSTM_ShouldHandleVariousBatchSizes(int batchSize)
    {
        var input = CreateTestData(batchSize, timeSteps: 30, features: 3);
        var (anomalies, errorSum) = _engine.RunInferenceUsingLSTM(input);
        Assert.Equal(batchSize, anomalies.Length);
    }

    [Fact]
    public void Constructor_ShouldThrowOnInvalidPath()
    {
        Assert.Throws<System.IO.FileNotFoundException>(() =>
            new InferenceEngine("invalid/path"));
    }

    private static double[,,] CreateTestData(int batchSize, int timeSteps, int features)
    {
        var input = new double[batchSize, timeSteps, features];
        var random = new Random(42);

        for (int i = 0; i < batchSize; i++)
            for (int j = 0; j < timeSteps; j++)
                for (int k = 0; k < features; k++)
                    input[i, j, k] = random.NextDouble();

        return input;
    }

    private static double[,] CreateTestData(int samples, int features)
    {
        var input = new double[samples, features];
        var random = new Random(42);

        for (int i = 0; i < samples; i++)
            for (int j = 0; j < features; j++)
                input[i, j] = random.NextDouble();

        return input;
    }

    public void Dispose()
    {
        _engine?.Dispose();
    }
}