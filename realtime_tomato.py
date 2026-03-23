import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 1. Load the TensorRT Engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "microvit_tomato_v15.engine"

with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 2. Setup GPU Memory Buffers
# Our input is 1x3x224x224 (float32)
h_input = cuda.pagelocked_empty(1 * 3 * 224 * 224, dtype=np.float32)
h_output = cuda.pagelocked_empty(5, dtype=np.float32) # 5 classes for tomato stages
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

# Labels from your original script
class_names = ['Stage 2', 'Stage 3', 'Stage 5', 'Stage 1', 'Stage 4']

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    # Normalization constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1)).ravel() # Flatten for GPU copy
    return img

cap = cv2.VideoCapture(0)
print("Starting Camera with TensorRT... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run preprocessing and copy to GPU
    h_input[:] = preprocess(frame)
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run Inference
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Copy back results
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # Calculate probabilities
    exp_logits = np.exp(h_output - np.max(h_output))
    probabilities = exp_logits / np.sum(exp_logits)
    
    pred = np.argmax(probabilities)
    conf = probabilities[pred] * 100

    # UI Logic
    label = class_names[pred] if conf > 50 else "Uncertain"
    text = f"{label} ({conf:.2f}%)"

    cv2.rectangle(frame, (10, 10), (550, 70), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Tomato Detection - Scientech AI (TensorRT)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
