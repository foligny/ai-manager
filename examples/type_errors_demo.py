"""
Demonstration of type errors that will be caught by the type-safe interface.

This file contains examples of incorrect usage that will be flagged by:
- IDE (PyCharm, VSCode, etc.)
- Type checker (mypy)
- Runtime validation (Pydantic)
"""

from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo


def demonstrate_type_errors():
    """This function contains examples of type errors that will be caught."""
    
    manager = AIManager(
        project_name="type_errors_demo",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    with manager.run(name="error_demo") as run:
        # ❌ TYPE ERROR: Invalid metric value types
        # These will be caught by IDE and type checker
        
        # Error: List not allowed as metric value
        run.log({
            "loss": [0.5, 0.4, 0.3],  # ❌ Type error: List not allowed
            "accuracy": {"train": 0.8, "val": 0.7},  # ❌ Type error: Dict not allowed
            "model": object(),  # ❌ Type error: Object not allowed
            "function": lambda x: x,  # ❌ Type error: Function not allowed
        })
        
        # Error: Invalid single metric value
        run.log_metric("loss", [0.5, 0.4])  # ❌ Type error: List not allowed
        run.log_metric("accuracy", {"train": 0.8})  # ❌ Type error: Dict not allowed
        
        # Error: Invalid configuration values
        run.log_config({
            "model": object(),  # ❌ Type error: Object not JSON serializable
            "function": lambda x: x,  # ❌ Type error: Function not allowed
            "complex": complex(1, 2),  # ❌ Type error: Complex not JSON serializable
        })
        
        # Error: Invalid artifact type
        run.log_artifact(ArtifactInfo(
            file_path="model.pth",
            artifact_type="invalid_type"  # ❌ Type error: Must be ArtifactType enum
        ))
        
        # Error: Invalid run status
        run.finish("invalid_status")  # ❌ Type error: Must be RunStatus enum


def demonstrate_correct_usage():
    """This function shows the correct usage that passes type checking."""
    
    manager = AIManager(
        project_name="type_errors_demo",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    with manager.run(name="correct_demo") as run:
        # ✅ CORRECT: Valid metric types
        run.log({
            "loss": 0.5,                    # ✅ float
            "accuracy": 0.85,               # ✅ float
            "epoch": 10,                    # ✅ int
            "is_training": True,            # ✅ bool
            "model_type": "transformer",    # ✅ str
            "learning_rate": 0.001          # ✅ float
        })
        
        # ✅ CORRECT: Valid single metric
        run.log_metric("loss", 0.5)        # ✅ float
        run.log_metric("accuracy", 0.85)   # ✅ float
        run.log_metric("epoch", 10)        # ✅ int
        
        # ✅ CORRECT: Valid configuration
        run.log_config({
            "batch_size": 32,               # ✅ int
            "learning_rate": 0.001,        # ✅ float
            "optimizer": "adam",            # ✅ str
            "use_gpu": True,                # ✅ bool
            "layers": [64, 128, 256],      # ✅ list (JSON serializable)
            "config": {"dropout": 0.1}     # ✅ dict (JSON serializable)
        })
        
        # ✅ CORRECT: Valid artifact
        run.log_artifact(ArtifactInfo(
            file_path="model.pth",
            name="best_model",
            artifact_type=ArtifactType.MODEL  # ✅ Valid enum
        ))
        
        # ✅ CORRECT: Valid run status
        run.finish(RunStatus.COMPLETED)     # ✅ Valid enum


if __name__ == "__main__":
    print("🔍 Type Safety Interface Demo")
    print("=" * 50)
    
    print("✅ The type-safe interface provides:")
    print("1. IDE autocomplete and error detection")
    print("2. Compile-time type checking")
    print("3. Runtime validation with Pydantic")
    print("4. Better code documentation")
    
    print("\n❌ Type errors that will be caught:")
    print("1. Invalid metric value types (lists, dicts, objects)")
    print("2. Non-JSON serializable configuration values")
    print("3. Invalid artifact types (must use ArtifactType enum)")
    print("4. Invalid run statuses (must use RunStatus enum)")
    
    print("\n🚀 Benefits:")
    print("- Catch errors before runtime")
    print("- Better IDE support with autocomplete")
    print("- Self-documenting code")
    print("- Consistent API usage")
    
    # Note: The incorrect examples are commented out to prevent actual errors
    # Uncomment demonstrate_type_errors() to see IDE/type checker errors
    # demonstrate_type_errors()
    
    # demonstrate_correct_usage()  # This would work correctly 