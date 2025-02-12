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
        public void TestClassicalModelInference()
        {
            System.Diagnostics.Debug.WriteLine("TestClassicalModelInference");
            // Arrange
            string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\csv";
            string inputPath = Path.Combine(rootFolder, "classical.csv");

            FileHandler fileHandler = new();
            double[,] input = fileHandler.LoadData(inputPath);
            InferenceEngine inferenceEngine = new();

            // Act
            bool[] result = inferenceEngine.RunInferenceUsingClassicalModel(input);

            // Assert
            System.Diagnostics.Debug.WriteLine("TestClassicalModelInference Complete");
        }

        [Fact]
        public void TestLSTMModelInference()
        {
            System.Diagnostics.Debug.WriteLine("TestLSTMModelInference");
            // Arrange
            string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\csv";
            string inputPath = Path.Combine(rootFolder, "data.csv");
            FileHandler fileHandler = new();
            double[,] input = fileHandler.LoadData(inputPath);

            DataProcessor dataProcessor = new();
            double[,,] lstmInput = dataProcessor.GetLSTMModelInput(input);

            InferenceEngine inferenceEngine = new();
            
            // Act
            bool[] result = inferenceEngine.RunInferenceUsingLSTM(lstmInput);

            // Assert
            System.Diagnostics.Debug.WriteLine("TestLSTMModelInference Complete");
        }
    }
}
