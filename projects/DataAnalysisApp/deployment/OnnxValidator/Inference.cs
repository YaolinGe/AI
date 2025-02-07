//namespace OnnxValidator;

//// InferenceEngine.cs
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using System;
//using System.Collections.Generic;
//using System.Linq;

//public class InferenceEngine : IDisposable
//{
//    private readonly DataProcessor _dataProcessor;
//    private readonly InferenceSession _classicalModel;
//    private readonly InferenceSession _lstmModel;
    
//    public InferenceEngine(DataProcessor dataProcessor)
//    {
//        _dataProcessor = dataProcessor;
//        _classicalModel = new InferenceSession("GNB.onnx");
//        _lstmModel = new InferenceSession("LSTM.onnx");
//    }

//    public (float[] ClassicalResults, float[] LstmResults) RunInference(DataInput rawData)
//    {
//        var processed = _dataProcessor.Process(rawData);
        
//        return (
//            PredictClassical(processed.Classical.Test),
//            PredictLSTM(processed.LSTM.Test)
//        );
//    }

//    private float[] PredictClassical(double[][] testData)
//    {
//        var inputTensor = new DenseTensor<float>(
//            testData.SelectMany(row => row.Select(v => (float)v)).ToArray(),
//            new[] { testData.Length, testData[0].Length });

//        var inputs = new List<NamedOnnxValue>
//        {
//            NamedOnnxValue.CreateFromTensor(_classicalModel.InputNames[0], inputTensor)
//        };

//        using var results = _classicalModel.Run(inputs);
//        return results.First().AsTensor<float>().ToArray();
//    }

//    private float[] PredictLSTM(double[][][] testData)
//    {
//        var inputTensor = new DenseTensor<float>(
//            testData.SelectMany(seq => seq.SelectMany(f => f)).Select(v => (float)v).ToArray(),
//            new[] { testData.Length, testData[0].Length, testData[0][0].Length });

//        var inputs = new List<NamedOnnxValue>
//        {
//            NamedOnnxValue.CreateFromTensor(_lstmModel.InputNames[0], inputTensor)
//        };

//        using var results = _lstmModel.Run(inputs);
//        return results.First().AsTensor<float>().ToArray();
//    }

//    public void Dispose()
//    {
//        _classicalModel.Dispose();
//        _lstmModel.Dispose();
//    }
//}