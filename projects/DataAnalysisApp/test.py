import numpy as np
import onnxruntime as ort
import time

def test_onnx_gpu():
    # Print ONNX Runtime version
    # print(f"ONNX Runtime version: {ort.__version__}")
    
    # Check available providers
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Check if CUDA provider is available
    if 'CUDAExecutionProvider' not in providers:
        print("ERROR: CUDA provider not available in ONNX Runtime!")
        return False
    
    # Create a simple session with CUDA as the preferred provider
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 0  # Enable verbose logging
    
    try:
        # Create a simple model to test (a matrix multiplication)
        # Input: 1x3 matrix, Output: 1x2 matrix
        # The model will multiply input by a 3x2 weight matrix
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create a session with CUDA provider
        print("Creating session with CUDA provider...")
        session = ort.InferenceSession("test_model.onnx", 
                                       providers=providers,
                                       sess_options=session_options)
        
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create a sample input
        input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        # Warm-up run
        session.run([output_name], {input_name: input_data})
        
        # Benchmark
        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            session.run([output_name], {input_name: input_data})
        end_time = time.time()
        
        print(f"Successfully ran inference on GPU!")
        print(f"Average inference time: {(end_time - start_time) / iterations * 1000:.2f} ms")
        print(f"Provider used: {session.get_providers()[0]}")
        return True
    
    except Exception as e:
        print(f"Error during ONNX Runtime GPU testing: {e}")
        return False

def create_test_model():
    """Create a simple ONNX model for testing"""
    import onnx
    from onnx import helper, TensorProto
    from onnx import numpy_helper
    
    # Create a simple model (Y = X * W)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
    
    # Create weight tensor (3x2 matrix)
    weight_value = np.array([[1.0, 2.0], 
                             [3.0, 4.0], 
                             [5.0, 6.0]], dtype=np.float32)
    weight = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['W'],
        value=helper.make_tensor(
            name='W_value',
            data_type=TensorProto.FLOAT,
            dims=weight_value.shape,
            vals=weight_value.flatten().tolist()
        )
    )
    
    # MatMul node
    matmul = helper.make_node('MatMul', ['X', 'W'], ['Y'])
    
    # Create the graph and model
    graph = helper.make_graph(
        nodes=[weight, matmul],
        name='simple_matmul',
        inputs=[X],
        outputs=[Y]
    )
    
    model = helper.make_model(graph, producer_name='onnx-gpu-test')
    onnx.checker.check_model(model)
    
    # Save the model
    onnx.save(model, 'test_model.onnx')
    print("Test ONNX model created successfully")

if __name__ == "__main__":
    print("ONNX Runtime GPU Test")
    print("-" * 50)
    
    # Create test model
    create_test_model()
    
    # Run GPU test
    success = test_onnx_gpu()
    
    if success:
        print("\nSUCCESS: ONNX Runtime can run on GPU!")
    else:
        print("\nFAILURE: ONNX Runtime could not run on GPU!")
        print("\nTroubleshooting tips:")
        print("1. Check CUDA version compatibility (run 'nvidia-smi' to see installed version)")
        print("2. Check cuDNN version compatibility")
        print("3. Verify that the ONNX Runtime was built with CUDA support")
        print("4. Make sure your GPU drivers are up to date")