# 🚀 AI Manager Training Guide

## Quick Start Commands

### Start the Server
```bash
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Dashboard
- **URL:** http://localhost:8000
- **Login:** admin / admin123

## 📊 Viewing Training Performance

### 1. Web Dashboard
1. Go to http://localhost:8000
2. Login with admin/admin123
3. Select a project
4. Click on a run to see metrics
5. View real-time training curves

### 2. Performance Analysis
```bash
# Show all available runs
python performance_analysis.py

# Quick performance overview
python quick_performance_demo.py
```

### 3. Key Metrics to Watch
- **Training Loss** → Should decrease
- **Validation Loss** → Should decrease (watch for overfitting)
- **Training Accuracy** → Should increase
- **Validation Accuracy** → Should increase

## 🔄 Resetting Training

### Delete Specific Runs
```bash
# Interactive cleanup
python targeted_cleanup.py

# Delete test/debug runs
python project_cleanup.py
```

### Fresh Start
```bash
# Stop server (Ctrl+C)
# Delete database
rm ai_manager.db

# Restart server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📈 Understanding Curves

### ✅ Good Training
- Loss decreasing smoothly
- Accuracy increasing
- Train/validation curves close
- No oscillation

### ⚠️ Warning Signs
- Validation loss increasing → Overfitting
- Training loss not decreasing → LR too low
- Large gap train/validation → Overfitting
- Oscillating curves → LR too high

## 🎯 Best Practices

### Before Training
- [ ] Set up project in dashboard
- [ ] Configure hyperparameters
- [ ] Prepare data pipeline
- [ ] Set up monitoring

### During Training
- [ ] Monitor curves every few epochs
- [ ] Check for overfitting
- [ ] Save best models
- [ ] Document changes

### After Training
- [ ] Compare with baselines
- [ ] Export results
- [ ] Save artifacts
- [ ] Document findings

## 🔧 Troubleshooting

### Training Not Improving
- Increase learning rate
- Check data quality
- Verify model architecture

### Overfitting
- Add regularization
- Reduce model complexity
- Use data augmentation
- Early stopping

### Unstable Training
- Reduce learning rate
- Increase batch size
- Check data normalization

## 📋 Quick Reference

| Action | Command |
|--------|---------|
| Start server | `python -m uvicorn app.main:app --reload` |
| View dashboard | http://localhost:8000 |
| Analyze runs | `python performance_analysis.py` |
| Clean up runs | `python targeted_cleanup.py` |
| Reset everything | `rm ai_manager.db && restart server` |

## 🎯 Training Integration Example

```python
from ai_manager import AIManager

# Initialize
manager = AIManager(project_name="my_project")

# Start run
with manager.run(name="experiment_1") as run:
    # Log config
    run.config.update({
        "learning_rate": 0.001,
        "batch_size": 32
    })
    
    # Training loop
    for epoch in range(100):
        # Your training code
        train_loss = train_epoch()
        val_loss = validate()
        
        # Log metrics
        run.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
```

## 📞 Need Help?

1. **Check server logs** for errors
2. **Verify database** is accessible
3. **Check network** connectivity
4. **Review configuration** files
5. **Restart server** if needed

---

**Happy Training! 🎉** 