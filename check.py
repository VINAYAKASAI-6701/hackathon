import subprocess
import sys

# Make sure scripts match actual file names in hackathon\hackathon
scripts = [
    "DATA_PROCESSING..py",   # keep the double dots if your file really has it
    "MODEL_TRAINING.py",
    "blockchain.py",
    "APP.py"
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Script {script} failed with exit code {e.returncode}")
