# AI Manager Testing Guide

This guide explains how to test the AI Manager with sample inputs and see outputs on screen.

## 🚀 Quick Start

### 1. Start the Server
```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test the AI Agent

#### **Option A: Automated Test (Recommended)**
```bash
python test_ai_agent.py
```
This script will:
- ✅ Login automatically
- ✅ Create a demo project with tags
- ✅ Create multiple runs with different configurations
- ✅ Simulate training with realistic metrics
- ✅ Show all results on screen

#### **Option B: Interactive Test**
```bash
python interactive_test.py
```
This provides an interactive menu where you can:
- 📁 Create your own projects
- 🏃 Create your own runs
- 📊 Log custom metrics
- 🎯 Simulate training with your parameters

#### **Option C: Enhance Existing Demo**
```bash
python enhance_demo.py
```
This will add tags to your existing demo projects and create additional demo runs.

## 🎯 What You'll See

### **Automated Test Output:**
```
🤖 AI Manager Test Agent
==================================================

1️⃣ Testing Login...
✅ Logged in successfully as admin

2️⃣ Creating Demo Project with Tags...
✅ Created project: demo_image_classifier (ID: 3)
   Tags: demo, image-classification, cnn, pytorch, computer-vision

3️⃣ Creating Test Runs with Tags...
✅ Created run: baseline_cnn (ID: 25)
   Tags: baseline, cnn, small-model
✅ Created run: large_resnet (ID: 26)
   Tags: large-model, resnet, high-accuracy
✅ Created run: experiment_attention (ID: 27)
   Tags: experiment, attention, research

4️⃣ Simulating Training...
🎯 Simulating training for run 25...
✅ Epoch 0: loss=1.9000, acc=0.3000
✅ Epoch 1: loss=1.7100, acc=0.3600
✅ Epoch 2: loss=1.5390, acc=0.4200
✅ Training simulation completed for run 25

5️⃣ Listing All Projects and Runs...
📁 Found 2 projects:
   • demo_image_classifier (ID: 3) [demo, image-classification, cnn, pytorch, computer-vision]
   • sample_image_classifier (ID: 4) [demo, image-classification, cnn, pytorch, computer-vision]

🏃 Found 3 runs for project 3:
   • baseline_cnn (ID: 25) - completed [baseline, cnn, small-model]
   • large_resnet (ID: 26) - completed [large-model, resnet, high-accuracy]
   • experiment_attention (ID: 27) - completed [experiment, attention, research]

✅ Test completed! Check the web dashboard at http://localhost:8000
```

### **Interactive Test Menu:**
```
🎯 Interactive AI Manager Test Menu
==================================================
1. List all projects
2. Create new project
3. List runs for a project
4. Create new run
5. Log metrics manually
6. Simulate training
7. Exit
--------------------------------------------------
Choose an option (1-7): 2

📁 Create New Project
------------------------------
Project name: my_test_project
Description (optional): Testing image classification
Enter tags (comma-separated, e.g., demo,cnn,pytorch): test,cnn,experiment
Tags: test, cnn, experiment
✅ Created project: my_test_project (ID: 5)
   Tags: test, cnn, experiment
```

## 🏷️ Tag System

### **Project Tags:**
- `demo` - Demo projects
- `image-classification` - Computer vision projects
- `nlp` - Natural language processing
- `computer-vision` - Image/video processing
- `pytorch` - PyTorch framework
- `tensorflow` - TensorFlow framework
- `research` - Research experiments
- `production` - Production models

### **Run Tags:**
- `baseline` - Baseline models
- `experiment` - Experimental runs
- `small-model` - Lightweight models
- `large-model` - Complex models
- `high-accuracy` - High-performance runs
- `fast-training` - Quick training runs
- `attention` - Attention mechanisms
- `resnet` - ResNet architectures

## 📊 Testing Metrics

### **Manual Metric Logging:**
```
📊 Log Metrics for Run 25
-----------------------------------
Enter metrics (or 'done' to finish):
Format: metric_name=value (e.g., loss=0.5, accuracy=0.85)
Metrics: loss=0.45, accuracy=0.92, f1_score=0.89
Step (default: 0): 10
✅ Logged metrics for step 10: {'loss': 0.45, 'accuracy': 0.92, 'f1_score': 0.89}
```

### **Simulated Training:**
```
🎯 Simulate Training for Run 25
----------------------------------------
Number of epochs (default: 10): 5
Delay between epochs in seconds (default: 0.5): 0.3

🎯 Simulating 5 epochs with 0.3s delay...
✅ Epoch 0: loss=1.9000, acc=0.3000
✅ Epoch 1: loss=1.7100, acc=0.3600
✅ Epoch 2: loss=1.5390, acc=0.4200
✅ Epoch 3: loss=1.3851, acc=0.4800
✅ Epoch 4: loss=1.2466, acc=0.5400
✅ Training simulation completed for run 25
```

## 🌐 Web Dashboard

After running the tests, visit **http://localhost:8000** to see:

### **Projects with Tags:**
- 📁 **demo_image_classifier** `[demo, image-classification, cnn, pytorch, computer-vision]`
- 📁 **sample_image_classifier** `[demo, image-classification, cnn, pytorch, computer-vision]`

### **Runs with Tags:**
- 🏃 **baseline_cnn** `[baseline, cnn, small-model]` - completed
- 🏃 **large_resnet** `[large-model, resnet, high-accuracy]` - completed
- 🏃 **experiment_attention** `[experiment, attention, research]` - completed

### **Real-time Metrics:**
- 📊 Training curves updating live
- 📈 Loss and accuracy plots
- 🔄 WebSocket updates

## 🔧 Custom Testing

### **Create Your Own Test:**
```python
from test_ai_agent import AIManagerTestAgent

# Initialize agent
agent = AIManagerTestAgent()

# Login
agent.login()

# Create custom project
project = agent.create_test_project(
    name="my_custom_project",
    description="My custom AI project",
    tags=["custom", "neural-network", "research"]
)

# Create custom run
run = agent.create_test_run(
    project_id=project['id'],
    name="my_experiment",
    tags=["experiment", "attention", "transformer"]
)

# Simulate training
agent.simulate_training(run['id'], epochs=20)
```

### **Test Different Scenarios:**
1. **Image Classification**: Use tags like `cnn`, `resnet`, `computer-vision`
2. **NLP Projects**: Use tags like `nlp`, `transformer`, `bert`
3. **Research**: Use tags like `research`, `experiment`, `attention`
4. **Production**: Use tags like `production`, `optimized`, `deployed`

## 🐛 Troubleshooting

### **Common Issues:**

1. **Connection Error:**
   ```
   ❌ Connection error: Connection refused
   ```
   **Solution:** Make sure the server is running on port 8000

2. **Login Failed:**
   ```
   ❌ Login failed: 401
   ```
   **Solution:** Use username `admin` and password `admin123`

3. **Project Creation Failed:**
   ```
   ❌ Failed to create project: 400
   ```
   **Solution:** Project name already exists, try a different name

4. **Metrics Not Updating:**
   ```
   ❌ Failed to log metrics: 404
   ```
   **Solution:** Make sure the run ID exists and is valid

### **Debug Mode:**
Add debug output to see detailed API calls:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Next Steps

1. **Explore the Web Dashboard** - See your projects and runs with tags
2. **Try Interactive Testing** - Create your own custom experiments
3. **Add Real Training** - Integrate with your actual ML training code
4. **Custom Metrics** - Log your own custom metrics and visualizations

Happy testing! 🚀 