using Xunit;
using OnnxValidator;
using System.Collections.Generic;
using System.IO;

namespace TestOnnxPipeline
{
    public class UnitTestInferenceEngine
    {


        public UnitTestInferenceEngine()
        {
            //InferenceEngine inferenceEngine = new();

        }

        [Fact]
        public void TestInferenceEngine()
        {
            System.Diagnostics.Debug.WriteLine("TestInferenceEngine");
            // Arrange
            string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\csv";
            string inputPath = Path.Combine(rootFolder, "output.csv");

            FileHandler fileHandler = new();
            double[,] input = fileHandler.LoadData(inputPath);
            InferenceEngine inferenceEngine = new();

            // Act
            inferenceEngine.RunInference(input);

            // Assert
            System.Diagnostics.Debug.WriteLine("TestInferenceEngine Complete");
        }
    }
}
