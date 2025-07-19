#!/usr/bin/env python3
"""
HTTPS development server for AI Manager.
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    cert_file = Path("certs/cert.pem")
    key_file = Path("certs/key.pem")
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ SSL certificates not found. Run setup_https.py first.")
        sys.exit(1)
    
    print("🚀 Starting AI Manager with HTTPS...")
    print("   URL: https://localhost:8000")
    print("   Note: You may see a security warning - this is normal for self-signed certificates")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile=str(key_file),
        ssl_certfile=str(cert_file)
    )
