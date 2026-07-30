import subprocess
import time
import sys

# Start FastAPI
fastapi = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000"]
)

# Wait a moment for FastAPI to start
time.sleep(2)

try:
    # Start Streamlit
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"]
    )
finally:
    # Stop FastAPI when Streamlit exits
    fastapi.terminate()
    fastapi.wait()