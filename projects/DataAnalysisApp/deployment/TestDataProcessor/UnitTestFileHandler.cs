using Xunit;
using OnnxValidator;
using System.Collections.Generic;
using System.IO;

namespace TestOnnxPipeline
{
    public class UnitTestFileHandler
    {
        string rootFolder;
        string dataPath;
        string minMaxPath;

        public UnitTestFileHandler()
        {
            rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator\csv";
            dataPath = Path.Combine(rootFolder, "data.csv");
            minMaxPath = Path.Combine(rootFolder, "min_max_values.csv");
        }

        [Fact]
        public void TestLoadData()
        {
            System.Diagnostics.Debug.WriteLine("TestLoadData");

            // Arrange
            FileHandler fileHandler = new();

            // Act
            double[,] data = fileHandler.LoadData(dataPath);

            // Assert
            Assert.NotNull(data);
        }

        [Fact]
        public void TestLoadMinMaxValues()
        {
            System.Diagnostics.Debug.WriteLine("TestLoadMinMaxValues");
            // Arrange
            FileHandler fileHandler = new();

            // Act
            Dictionary<int, (double min, double max)> customRanges = fileHandler.LoadMinMaxValues(minMaxPath);

            // Assert
            Assert.NotNull(customRanges);
        }
    }
}
