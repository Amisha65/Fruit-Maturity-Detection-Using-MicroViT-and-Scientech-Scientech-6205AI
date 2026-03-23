# Real-Time Fruit Maturity Detection using MicroViT-S1

## Abstract

This work presents a real-time fruit maturity detection system based on **MicroViT-S1**, a lightweight Vision Transformer designed for efficient inference on edge devices. The proposed system aims to provide accurate maturity-stage classification with low latency and high throughput, making it suitable for practical deployment in resource-constrained environments.

The current experimental evaluation is conducted on a **tomato maturity dataset**, where the model classifies fruits into **five maturity stages**. Although the present implementation is validated on tomatoes, the framework is designed for broader **fruit maturity detection tasks** and can be extended to other fruits in future work.

Experimental results show that the proposed model achieves **99.1% classification accuracy** and **61.7 FPS** after TensorRT optimization, demonstrating its effectiveness for real-time edge deployment.

## 1. Introduction

Fruit maturity detection is an important task in precision agriculture, automated grading, and post-harvest quality assessment. Traditional manual inspection methods are often subjective, time-consuming, and difficult to scale. Deep learning-based vision systems provide an effective alternative; however, many high-performing models are computationally expensive and unsuitable for edge deployment.

To address this challenge, this project proposes **MicroViT-S1**, a lightweight Vision Transformer architecture optimized for real-time inference on embedded hardware. The system is intended not only for agricultural monitoring and automated sorting, but also for future integration into **agricultural robots** and assistive systems for **visually impaired individuals**.

## 2. Objective

The objective of this work is to develop a real-time fruit maturity detection system that:

- accurately classifies maturity stages
- operates efficiently on edge hardware
- maintains low latency and high throughput
- supports future extension to multiple fruit categories

At the current stage, the system is trained and evaluated on a tomato maturity dataset as the initial benchmark.

## 3. Proposed Method

### 3.1 MicroViT-S1

The core contribution of this work is **MicroViT-S1**, a lightweight Vision Transformer developed for efficient edge inference.

#### Key Features

- Hybrid **CNN + Transformer** architecture
- Efficient spatial attention mechanism
- Low parameter count for faster inference
- Designed specifically for real-time applications

Unlike conventional CNN-based models, MicroViT-S1 captures both local and global contextual information, improving classification performance for visually similar maturity stages.

### 3.2 Inference Pipeline

```text
PyTorch Model (.pth)
        ↓
ONNX Conversion (.onnx)
        ↓
TensorRT Optimization (.engine)
        ↓
Real-Time Inference (OpenCV)
```

## 4. Edge Deployment Platform

The proposed system was deployed on a **Scientech 6205AI Artificial Intelligence Workstation**, based on the **NVIDIA Jetson Nano (T210 architecture)**, to evaluate real-time inference on resource-constrained edge hardware.

### Configuration

- **GPU:** 128-core NVIDIA Maxwell GPU
- **CPU:** Quad-core ARM Cortex-A57 @ 1.43 GHz
- **RAM:** 4 GB LPDDR4
- **OS:** Ubuntu 18.04 (JetPack R32.7.6)
- **TensorRT:** 8.2.1
- **Frameworks:** PyTorch, ONNX, TensorRT, OpenCV

The device was used to deploy the optimized **MicroViT-S1 TensorRT engine** and run real-time inference using live camera input.

## 5. Dataset Description

The dataset used in the current study consists of tomato images categorized into five maturity stages.

### Dataset Summary

- **Total Images:** 120
- **Number of Classes:** 5
- **Images per Class:** 24

### Classes

- Stage 1 - Unripe
- Stage 2
- Stage 3
- Stage 4
- Stage 5 - Fully Ripe

Although the present dataset is tomato-specific, the framework is intended for broader fruit maturity detection in future work.

## 6. Experimental Results

### 6.1 Classification Performance

| Stage | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Stage 1 | 1.00 | 1.00 | 1.00 |
| Stage 2 | 0.96 | 1.00 | 0.98 |
| Stage 3 | 1.00 | 1.00 | 1.00 |
| Stage 4 | 1.00 | 0.96 | 0.98 |
| Stage 5 | 1.00 | 1.00 | 1.00 |

**Overall Accuracy:** **99.1%**

### 6.2 TensorRT Performance

| Metric | Value |
|--------|-------|
| Throughput | 61.7 FPS |
| Mean Latency | 16.14 ms |
| GPU Compute Time | 16.00 ms |

### 6.3 Performance Analysis

The optimized TensorRT pipeline demonstrates:

- minimal inference overhead
- efficient GPU utilization
- suitability for real-time deployment on edge devices

The system achieves **more than 60 FPS**, making it suitable for practical agricultural applications.

## 7. Model Comparison

| Model | Accuracy | FPS | Latency |
|-------|----------|-----|---------|
| MicroViT-S1 | 99.1% | 61.7 | 16.14 ms |
| EfficientNetV2-B0 | 99.0% | 48.3 | 20.6 ms |

The comparison shows that **MicroViT-S1** delivers better speed with comparable accuracy, making it more suitable for real-time edge deployment.

## 8. Evaluation Metrics

The model was evaluated using the following metrics:

- Precision
- Recall
- F1-score
- Accuracy

## 9. Applications

The proposed system has potential applications in:

- fruit maturity detection in agriculture
- automated sorting and grading systems
- agricultural robots for harvesting and monitoring
- assistive intelligent systems for visually impaired individuals

## 10. Implementation Details

### Tech Stack

- Python
- PyTorch
- ONNX
- NVIDIA TensorRT
- CUDA
- OpenCV

### How to Run

#### Step 1: Convert ONNX to TensorRT

```bash
trtexec --onnx=microvit_tomato.onnx --saveEngine=microvit.engine --fp16
```

#### Step 2: Run Real-Time Inference

```bash
python3 realtime_tomato.py
```

## 11. Conclusion

This project demonstrates that **MicroViT-S1** can achieve both high accuracy and real-time performance for fruit maturity detection on edge devices. On the current tomato maturity dataset, the system achieves:

- **99.1% accuracy**
- **61.7 FPS throughput**
- **16.14 ms latency**

These results indicate that lightweight Vision Transformers can effectively bridge the gap between deep learning research and real-world deployment.

## 12. Future Work

Future work may include:

- expanding the dataset to other fruits
- increasing dataset size for better generalization
- integration with automated sorting systems
- deployment on agricultural robots
- support for assistive systems for visually impaired people
- INT8 quantization for further acceleration
- multi-camera deployment

## 13. Project Structure

```text
project/
├── models/
│   ├── microvit.engine
│   ├── microvit.onnx
├── dataset/
├── realtime_tomato.py
├── evaluate_research.py
├── notebooks/
└── README.md
```

## 14. Repository

GitHub Repository:  
[https://github.com/Amisha65/Fruit-Maturity-Detection-Using-MicroViT-and-Scientech-Scientech-6205AI.git](https://github.com/Amisha65/Fruit-Maturity-Detection-Using-MicroViT-and-Scientech-Scientech-6205AI.git)
