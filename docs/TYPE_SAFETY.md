# Type-Safe Interface for AI Manager

The AI Manager now provides a **type-safe interface** that catches errors at development time using static type checking.

## 🎯 **Benefits**

- **IDE Support**: Autocomplete and error detection in PyCharm, VSCode, etc.
- **Compile-time Checking**: Catch errors before runtime with mypy
- **Runtime Validation**: Pydantic validation for data integrity
- **Self-documenting**: Clear interface requirements

## 📋 **Required Interface**

### **1. Metric Logging Interface**

```python
from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo

# Initialize manager
manager = AIManager(
    project_name="my_project",
    api_url="http://localhost:8000",
    username="admin",
    password="admin123"
)

# Start a run
with manager.run(name="experiment_1") as run:
    # ✅ CORRECT: Valid metric types
    run.log({
        "loss": 0.5,                    # float
        "accuracy": 0.85,               # float
        "epoch": 10,                    # int
        "is_training": True,            # bool
        "model_type": "transformer"     # str
    })
    
    # ✅ CORRECT: Single metric
    run.log_metric("learning_rate", 0.001)
    
    # ✅ CORRECT: Configuration
    run.log_config({
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.001
    })
    
    # ✅ CORRECT: Artifact with type safety
    artifact = ArtifactInfo(
        file_path="model.pth",
        name="best_model",
        artifact_type=ArtifactType.MODEL
    )
    run.log_artifact(artifact)
    
    # ✅ CORRECT: Finish with enum
    run.finish(RunStatus.COMPLETED)
```

### **2. Valid Data Types**

#### **Metrics** (`run.log()` and `run.log_metric()`)
```python
# ✅ Allowed types:
run.log({
    "loss": 0.5,                    # float
    "accuracy": 0.85,               # float
    "epoch": 10,                    # int
    "is_training": True,            # bool
    "model_type": "transformer"     # str
})

# ❌ NOT allowed:
run.log({
    "loss": [0.5, 0.4],            # List not allowed
    "accuracy": {"train": 0.8},     # Dict not allowed
    "model": object(),              # Object not allowed
    "function": lambda x: x         # Function not allowed
})
```

#### **Configuration** (`run.log_config()`)
```python
# ✅ Allowed types (JSON serializable):
run.log_config({
    "batch_size": 32,               # int
    "learning_rate": 0.001,        # float
    "optimizer": "adam",            # str
    "use_gpu": True,                # bool
    "layers": [64, 128, 256],      # list
    "config": {"dropout": 0.1}     # dict
})

# ❌ NOT allowed:
run.log_config({
    "model": object(),              # Object not JSON serializable
    "function": lambda x: x,        # Function not allowed
    "complex": complex(1, 2)        # Complex not JSON serializable
})
```

#### **Artifacts** (`run.log_artifact()`)
```python
from ai_manager import ArtifactInfo, ArtifactType

# ✅ CORRECT: Use ArtifactInfo with enum
artifact = ArtifactInfo(
    file_path="model.pth",
    name="best_model",
    artifact_type=ArtifactType.MODEL  # Must use enum
)
run.log_artifact(artifact)

# ❌ NOT allowed:
run.log_artifact(ArtifactInfo(
    file_path="model.pth",
    artifact_type="invalid_type"    # String not allowed
))
```

#### **Run Status** (`run.finish()`)
```python
from ai_manager import RunStatus

# ✅ CORRECT: Use enum
run.finish(RunStatus.COMPLETED)
run.finish(RunStatus.FAILED)
run.finish(RunStatus.STOPPED)

# ❌ NOT allowed:
run.finish("completed")             # String not allowed
run.finish("invalid_status")        # Invalid string
```

## 🔧 **Type Checking Setup**

### **1. Install mypy**
```bash
pip install mypy types-requests
```

### **2. Run type checking**
```bash
# Check a specific file
mypy your_training_script.py

# Check entire project
mypy ai_manager/ examples/

# Check with strict settings
mypy --strict ai_manager/
```

### **3. IDE Integration**

#### **VS Code**
```json
{
    "python.linting.mypyEnabled": true,
    "python.linting.enabled": true
}
```

#### **PyCharm**
- Enable mypy in Settings → Tools → External Tools
- Or use the built-in type checker

## 🚨 **Error Examples**

### **Type Errors Caught by IDE/mypy:**

```python
# ❌ ERROR: Invalid metric value type
run.log({
    "loss": [0.5, 0.4],  # Type error: List not allowed
})

# ❌ ERROR: Invalid configuration
run.log_config({
    "model": object(),  # Type error: Object not JSON serializable
})

# ❌ ERROR: Invalid artifact type
run.log_artifact(ArtifactInfo(
    artifact_type="invalid"  # Type error: Must be ArtifactType enum
))

# ❌ ERROR: Invalid run status
run.finish("invalid")  # Type error: Must be RunStatus enum
```

### **Runtime Validation Errors:**

```python
# ❌ RUNTIME ERROR: Pydantic validation
run.log({
    "loss": complex(1, 2)  # Runtime error: Complex not allowed
})
```

## 📊 **Available Enums**

### **ArtifactType**
```python
from ai_manager import ArtifactType

ArtifactType.MODEL    # For model files
ArtifactType.DATA      # For datasets
ArtifactType.CONFIG    # For configuration files
ArtifactType.OTHER     # For other files
```

### **RunStatus**
```python
from ai_manager import RunStatus

RunStatus.RUNNING     # Training in progress
RunStatus.COMPLETED   # Training finished successfully
RunStatus.FAILED      # Training failed
RunStatus.STOPPED     # Training stopped manually
```

## 🎯 **Integration Examples**

### **PyTorch Training**
```python
from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo

manager = AIManager(project_name="pytorch_experiment")

with manager.run(name="transformer_training") as run:
    for epoch in range(num_epochs):
        # Your existing training code
        loss = train_epoch(model, dataloader, optimizer)
        accuracy = evaluate(model, val_dataloader)
        
        # ✅ Type-safe logging
        run.log({
            "loss": loss,           # float
            "accuracy": accuracy,    # float
            "epoch": epoch          # int
        })
    
    # ✅ Type-safe artifact logging
    torch.save(model.state_dict(), "model.pth")
    artifact = ArtifactInfo(
        file_path="model.pth",
        name="final_model",
        artifact_type=ArtifactType.MODEL
    )
    run.log_artifact(artifact)
    
    # ✅ Type-safe finish
    run.finish(RunStatus.COMPLETED)
```

### **TensorFlow/Keras Training**
```python
from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo

manager = AIManager(project_name="tensorflow_experiment")

with manager.run(name="cnn_training") as run:
    for epoch in range(num_epochs):
        # Your existing training code
        history = model.fit(train_dataset, validation_data=val_dataset)
        
        # ✅ Type-safe logging
        run.log({
            "loss": history.history['loss'][-1],           # float
            "val_loss": history.history['val_loss'][-1],   # float
            "accuracy": history.history['accuracy'][-1],   # float
            "epoch": epoch                                  # int
        })
    
    # ✅ Type-safe artifact logging
    model.save("model.h5")
    artifact = ArtifactInfo(
        file_path="model.h5",
        name="keras_model",
        artifact_type=ArtifactType.MODEL
    )
    run.log_artifact(artifact)
    
    # ✅ Type-safe finish
    run.finish(RunStatus.COMPLETED)
```

## 🔍 **Testing Type Safety**

### **1. Create test file with errors**
```python
# test_errors.py
from ai_manager import AIManager

manager = AIManager(project_name="test")
with manager.run() as run:
    # ❌ These will be caught by mypy
    run.log({"loss": [0.5, 0.4]})  # Type error
    run.finish("invalid")           # Type error
```

### **2. Run mypy**
```bash
mypy test_errors.py
# Output: Type errors detected!
```

## 🎉 **Summary**

The type-safe interface ensures:

1. **Compile-time error detection** with mypy
2. **IDE autocomplete and error highlighting**
3. **Runtime validation** with Pydantic
4. **Consistent API usage** across projects
5. **Self-documenting code** with clear type requirements

This prevents common mistakes like:
- Passing wrong data types to metrics
- Using invalid artifact types
- Using invalid run statuses
- Passing non-JSON serializable configuration

The interface is **backward compatible** but encourages type-safe usage for better development experience. 