import subprocess
import sys

scripts = [
    "dataprocessing.py",
    "modeltraining.py",
    "blockchain.py",
    "app.py"
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Script {script} failed with exit code {e.returncode}")
