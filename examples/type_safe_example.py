"""
Example demonstrating the type-safe interface of AI Manager.

This example shows how the type system catches errors at development time.
"""

from ai_manager import AIManager, RunStatus, ArtifactType, ArtifactInfo
from typing import Dict, Any


def correct_usage_example():
    """Example of correct usage that will pass type checking."""
    manager = AIManager(
        project_name="type_safe_example",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    with manager.run(name="correct_example") as run:
        # ✅ Correct: Logging valid metric types
        run.log({
            "loss": 0.5,                    # float
            "accuracy": 0.85,               # float
            "epoch": 10,                    # int
            "is_training": True,            # bool
            "model_type": "transformer"     # str
        })
        
        # ✅ Correct: Logging single metric
        run.log_metric("learning_rate", 0.001)
        
        # ✅ Correct: Logging configuration
        run.log_config({
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001
        })
        
        # ✅ Correct: Logging artifact with type-safe interface
        artifact = ArtifactInfo(
            file_path="model.pth",
            name="best_model",
            artifact_type=ArtifactType.MODEL
        )
        run.log_artifact(artifact)
        
        # ✅ Correct: Finishing with proper status
        run.finish(RunStatus.COMPLETED)


def incorrect_usage_examples():
    """Examples of incorrect usage that will be caught by type checking."""
    manager = AIManager(
        project_name="type_safe_example",
        api_url="http://localhost:8000",
        username="admin",
        password="admin123"
    )
    
    with manager.run(name="incorrect_example") as run:
        # ❌ Type Error: Invalid metric value type
        # run.log({
        #     "loss": [0.5, 0.4],  # List is not allowed
        #     "model": {"layers": 10},  # Dict is not allowed
        #     "complex_object": object()  # Object is not allowed
        # })
        
        # ❌ Type Error: Invalid metric value
        # run.log_metric("loss", [0.5, 0.4])  # List not allowed
        
        # ❌ Type Error: Invalid configuration value
        # run.log_config({
        #     "model": object(),  # Object not JSON serializable
        #     "complex": lambda x: x  # Function not allowed
        # })
        
        # ❌ Type Error: Invalid artifact type
        # run.log_artifact(ArtifactInfo(
        #     file_path="model.pth",
        #     artifact_type="invalid_type"  # Must be ArtifactType enum
        # ))
        
        # ❌ Type Error: Invalid run status
        # run.finish("invalid_status")  # Must be RunStatus enum
        
        pass


def type_checking_demo():
    """Demonstrate how type checking catches errors."""
    print("🔍 Type Safety Demo")
    print("=" * 50)
    
    print("✅ Correct usage examples:")
    print("- Valid metric types: float, int, bool, str")
    print("- Valid configuration: JSON serializable values")
    print("- Valid artifact types: ArtifactType enum")
    print("- Valid run status: RunStatus enum")
    
    print("\n❌ Type checking will catch:")
    print("- Invalid metric types (lists, dicts, objects)")
    print("- Non-JSON serializable configuration")
    print("- Invalid artifact types")
    print("- Invalid run statuses")
    
    print("\n🚀 Benefits:")
    print("- IDE autocomplete and error detection")
    print("- Compile-time error checking")
    print("- Runtime validation with Pydantic")
    print("- Better code documentation")


if __name__ == "__main__":
    type_checking_demo()
    
    # Uncomment to test correct usage (requires server running)
    # correct_usage_example()
    
    # Uncomment to see type errors (will be caught by IDE/type checker)
    # incorrect_usage_examples() 