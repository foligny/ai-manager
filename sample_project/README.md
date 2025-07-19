# Sample AI Project with AI Manager Integration

This is a complete sample AI project that demonstrates how to use the AI Manager for monitoring and logging training experiments. The project includes a small CNN model for image classification that can run efficiently on CPU.

## 🎯 **Project Overview**

- **Task**: Image classification (Circles, Squares, Triangles)
- **Model**: Small CNN (32x32 grayscale images, 3 classes)
- **Data**: 100 synthetic samples with realistic variations
- **Hardware**: CPU-optimized, no GPU required
- **Integration**: Full AI Manager monitoring and logging

## 📁 **Project Structure**

```
sample_project/
├── requirements.txt          # Dependencies
├── data_generator.py        # Synthetic data generation
├── model.py                 # SmallCNN model and trainer
├── train_with_ai_manager.py # Main training script
├── README.md               # This file
└── generated_files/        # Output files (created during training)
    ├── X.npy              # Training data
    ├── y.npy              # Labels
    ├── data_info.json     # Dataset information
    ├── sample_data.png    # Data visualization
    ├── model.pth          # Trained model
    ├── training_history.png # Training plots
    └── artifacts/         # AI Manager artifacts
```

## 🚀 **Quick Start**

### **1. Setup Environment**

```bash
# Navigate to sample project
cd sample_project

# Install dependencies
pip install -r requirements.txt

# Install AI Manager (from parent directory)
cd ..
pip install -e .
cd sample_project
```

### **2. Generate Data**

```bash
python data_generator.py
```

This will:
- Generate 100 synthetic images (circles, squares, triangles)
- Create train/test splits
- Save data files and visualization
- Display sample images

### **3. Test Model**

```bash
python model.py
```

This will:
- Create the SmallCNN model
- Test forward pass
- Display model information

### **4. Train with AI Manager**

```bash
# Single training run
python train_with_ai_manager.py --mode train

# Multiple runs with different configs
python train_with_ai_manager.py --mode retrain
```

## 📊 **Model Architecture**

### **SmallCNN Model**
```
Input: 32x32x1 (grayscale images)
├── Conv2d(1, 16, 3x3) + BatchNorm + ReLU + MaxPool2d
├── Conv2d(16, 32, 3x3) + BatchNorm + ReLU + MaxPool2d  
├── Conv2d(32, 64, 3x3) + BatchNorm + ReLU + MaxPool2d
├── Flatten
├── Linear(1024, 128) + ReLU + Dropout
├── Linear(128, 64) + ReLU + Dropout
└── Linear(64, 3) (output)
```

### **Model Specifications**
- **Parameters**: ~50K trainable parameters
- **Size**: ~0.2 MB
- **Memory**: < 100MB during training
- **Speed**: ~100 epochs/minute on CPU

## 📈 **Training Features**

### **AI Manager Integration**
- ✅ Real-time metric logging
- ✅ Configuration tracking
- ✅ Artifact management
- ✅ Training history visualization
- ✅ Multiple run comparison

### **Training Features**
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Batch normalization
- ✅ Dropout regularization
- ✅ Model checkpointing

### **Monitoring Metrics**
- ✅ Training/validation loss
- ✅ Training/validation accuracy
- ✅ Learning rate tracking
- ✅ Best model saving
- ✅ Final test evaluation

## 🎯 **Expected Results**

### **Training Performance**
- **Epochs**: 20-30 (with early stopping)
- **Final Accuracy**: 85-95%
- **Training Time**: 2-5 minutes
- **Memory Usage**: < 100MB

### **AI Manager Dashboard**
- Real-time training curves
- Model artifacts (checkpoints, plots)
- Configuration comparison
- Performance metrics

## 🔧 **Configuration Options**

### **Data Generation**
```python
# In data_generator.py
X, y = generate_synthetic_data(
    n_samples=100,      # Number of samples
    image_size=32,      # Image size
    n_classes=3,        # Number of classes
    noise_level=0.1     # Noise amount
)
```

### **Training Parameters**
```python
# In train_with_ai_manager.py
config = {
    "batch_size": 8,
    "learning_rate": 0.001,
    "max_epochs": 50,
    "early_stopping_patience": 10
}
```

### **Model Architecture**
```python
# In model.py
model = SmallCNN(
    num_classes=3,      # Number of output classes
    # Other parameters can be added here
)
```

## 📊 **Data Visualization**

The project generates several visualizations:

1. **Sample Data** (`sample_data.png`)
   - Shows examples of circles, squares, triangles
   - Demonstrates data variations

2. **Training History** (`training_history.png`)
   - Loss curves (train/validation)
   - Accuracy curves (train/validation)
   - Learning rate schedule
   - Final metrics comparison

## 🎯 **AI Manager Integration**

### **Real-time Monitoring**
```python
# Log metrics every epoch
run.log({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "train_accuracy": train_acc,
    "val_loss": val_loss,
    "val_accuracy": val_acc,
    "learning_rate": current_lr
})
```

### **Artifact Management**
```python
# Save and log model checkpoints
artifact = ArtifactInfo(
    file_path="best_model.pth",
    name="best_model_epoch_15",
    artifact_type=ArtifactType.MODEL,
    metadata={"epoch": 15, "val_accuracy": 0.92}
)
run.log_artifact(artifact)
```

### **Configuration Tracking**
```python
# Log experiment configuration
config = {
    "model_architecture": "SmallCNN",
    "learning_rate": 0.001,
    "batch_size": 8,
    "optimizer": "Adam"
}
run.log_config(config)
```

## 🔄 **Multiple Run Comparison**

The `--mode retrain` option demonstrates:

1. **Baseline Run**: Standard configuration
2. **High Learning Rate**: Faster learning
3. **Low LR + Large Batch**: Different optimization

Compare results in the AI Manager dashboard!

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Import Error**: Make sure AI Manager is installed
   ```bash
   cd .. && pip install -e . && cd sample_project
   ```

2. **CUDA Error**: Model runs on CPU by default
   ```python
   device = 'cpu'  # Already set in trainer
   ```

3. **Memory Error**: Reduce batch size
   ```python
   batch_size = 4  # Instead of 8
   ```

4. **Slow Training**: Model is designed for CPU
   - Training time: 2-5 minutes
   - Use early stopping for faster completion

### **Performance Tips**

1. **CPU Optimization**:
   - Model designed for CPU efficiency
   - Small batch sizes (4-8)
   - Early stopping enabled

2. **Memory Usage**:
   - < 100MB total memory
   - Automatic cleanup after training

3. **Training Speed**:
   - ~100 epochs/minute on modern CPU
   - Early stopping typically triggers at 20-30 epochs

## 📈 **Expected Training Curves**

### **Typical Results**
- **Epoch 1-5**: Rapid improvement (60% → 80%)
- **Epoch 6-15**: Steady improvement (80% → 90%)
- **Epoch 16-25**: Fine-tuning (90% → 95%)
- **Early Stopping**: Usually around epoch 20-30

### **Success Metrics**
- ✅ Final accuracy > 85%
- ✅ No overfitting (val acc close to train acc)
- ✅ Smooth learning curves
- ✅ Early stopping triggered

## 🎉 **Next Steps**

1. **Modify the Model**: Change architecture in `model.py`
2. **Add Data Augmentation**: Enhance `data_generator.py`
3. **Experiment with Hyperparameters**: Try different configs
4. **Extend to Real Data**: Replace synthetic data
5. **Add More Metrics**: Custom evaluation metrics

## 📚 **Learning Resources**

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **CNN Architecture**: Understanding convolutional networks
- **AI Manager Docs**: Type-safe interface documentation
- **Training Best Practices**: Early stopping, regularization

---

**Happy Training! 🚀**

This sample project demonstrates a complete AI workflow with professional monitoring and logging capabilities. 