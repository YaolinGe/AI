//namespace OnnxValidator;
//using System;
//using System.Collections.Generic;
//using System.Linq;

//public class Program
//{
//    public static void Main(string[] args)
//    {
//        string rootFolder = @"C:\Users\nq9093\CodeSpace\AI\projects\DataAnalysisApp\deployment\OnnxValidator";
//        string dataPath = Path.Combine(rootFolder, "data.csv");
//        string minMaxPath = Path.Combine(rootFolder, "min_max_values.csv");

//        // Create FileHandler instance
//        FileHandler fileHandler = new ();

//        // Load data
//        double[,] data = fileHandler.LoadData(dataPath);
//        Dictionary<int, (double min, double max)> customRanges = fileHandler.LoadMinMaxValues(minMaxPath);

//        // Create MinMaxScaler instance
//        MinMaxScaler scaler = new MinMaxScaler();
//        double[,] scaled = scaler.Transform(data, customRanges);

//        // // Calculate the first difference
//        //double[,] differenced = ComputeFirstDifference(scaled);

//        //InferenceEngine processor = new ();
//    }
//}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxGpuTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ONNX Runtime GPU Test in C#");
            Console.WriteLine("---------------------------");

            try
            {
                // Print ONNX Runtime version
                Console.WriteLine($"ONNX Runtime version: {typeof(InferenceSession).Assembly.GetName().Version}");

                // Check available providers
                var availableProviders = OrtEnv.Instance().GetAvailableProviders();
                Console.WriteLine($"Available providers: {string.Join(", ", availableProviders)}");

                // Check if CUDA provider is available
                if (!availableProviders.Contains("CUDAExecutionProvider"))
                {
                    Console.WriteLine("ERROR: CUDA provider not available in ONNX Runtime!");
                    PrintTroubleshootingTips();
                    return;
                }

                // Use the test ONNX model (create it using the Python script first)
                string modelPath = "test_model.onnx";
                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"ERROR: Model file {modelPath} not found!");
                    Console.WriteLine("Please run the Python script first to generate the test model.");
                    return;
                }

                // Set session options
                var sessionOptions = new SessionOptions();

                // Add the CUDA execution provider
                sessionOptions.AppendExecutionProvider_CUDA(); // Use default device ID (0)

                // Create a session with CUDA provider
                Console.WriteLine("Creating session with CUDA provider...");
                using var session = new InferenceSession(modelPath, sessionOptions);

                // Print active providers
                Console.WriteLine("Active providers: " +
                                  string.Join(", ", session.GetAvailableProviders()));

                // Create input tensor
                var inputData = new float[] { 1.0f, 2.0f, 3.0f };
                var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 3 });
                var inputs = new List<NamedOnnxValue> {
                    NamedOnnxValue.CreateFromTensor("X", inputTensor)
                };

                // Warm-up run
                session.Run(inputs);

                // Benchmark
                int iterations = 100;
                var stopwatch = new Stopwatch();
                stopwatch.Start();

                for (int i = 0; i < iterations; i++)
                {
                    using var results = session.Run(inputs);
                }

                stopwatch.Stop();
                double avgMs = stopwatch.ElapsedMilliseconds / (double)iterations;

                // Run once more to get and print the result
                using var finalResults = session.Run(inputs);
                var outputTensor = finalResults.First().AsTensor<float>();

                Console.WriteLine("Successfully ran inference!");
                Console.WriteLine($"Input: [{string.Join(", ", inputData)}]");
                Console.WriteLine($"Output: [{outputTensor.GetValue(0, 0)}, {outputTensor.GetValue(0, 1)}]");
                Console.WriteLine($"Average inference time: {avgMs:F2} ms");
                Console.WriteLine($"Provider used: {session.GetAvailableProviders().First()}");

                if (session.GetAvailableProviders().First() == "CUDAExecutionProvider")
                {
                    Console.WriteLine("\nSUCCESS: ONNX Runtime is running on GPU!");
                }
                else
                {
                    Console.WriteLine("\nWARNING: ONNX Runtime fell back to CPU execution!");
                    PrintTroubleshootingTips();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during ONNX Runtime GPU testing: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
                PrintTroubleshootingTips();
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }

        static void PrintTroubleshootingTips()
        {
            Console.WriteLine("\nTroubleshooting tips:");
            Console.WriteLine("1. Check if Microsoft.ML.OnnxRuntime.Gpu package is installed (not the CPU-only package)");
            Console.WriteLine("2. Check CUDA version compatibility (run 'nvidia-smi' to see installed version)");
            Console.WriteLine("3. Check cuDNN version compatibility");
            Console.WriteLine("4. Make sure your GPU drivers are up to date");
            Console.WriteLine("5. Check system environment PATH includes CUDA directories");
            Console.WriteLine("6. For Blazor, ensure WASM/WASI compatibility with GPU acceleration");
        }
    }
}