#!/usr/bin/env python3
"""
Setup script to generate self-signed SSL certificates for HTTPS development.
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_ssl_certificates():
    """Generate self-signed SSL certificates for development."""
    
    # Create certs directory
    certs_dir = Path("certs")
    certs_dir.mkdir(exist_ok=True)
    
    # Check if certificates already exist
    cert_file = certs_dir / "cert.pem"
    key_file = certs_dir / "key.pem"
    
    if cert_file.exists() and key_file.exists():
        print("✅ SSL certificates already exist")
        return True
    
    print("🔐 Generating self-signed SSL certificates...")
    
    try:
        # Generate private key
        subprocess.run([
            "openssl", "genrsa", "-out", str(key_file), "2048"
        ], check=True, capture_output=True)
        
        # Generate certificate
        subprocess.run([
            "openssl", "req", "-new", "-x509", "-key", str(key_file),
            "-out", str(cert_file), "-days", "365", "-subj",
            "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        ], check=True, capture_output=True)
        
        print("✅ SSL certificates generated successfully!")
        print(f"   Certificate: {cert_file}")
        print(f"   Private Key: {key_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating certificates: {e}")
        return False
    except FileNotFoundError:
        print("❌ OpenSSL not found. Please install OpenSSL:")
        print("   Ubuntu/Debian: sudo apt-get install openssl")
        print("   macOS: brew install openssl")
        return False

def create_https_server_script():
    """Create a script to run the server with HTTPS."""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    with open("run_https.py", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("run_https.py", 0o755)
    print("✅ Created run_https.py script")

if __name__ == "__main__":
    print("🔧 Setting up HTTPS for AI Manager development...")
    
    if generate_ssl_certificates():
        create_https_server_script()
        print("\n🎉 Setup complete!")
        print("\nTo start the server with HTTPS:")
        print("   python3 run_https.py")
        print("\nOr manually:")
        print("   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1) 