import os
import sys
import subprocess
import time


def start_streamlit():
    os.environ['STREAMLIT_SERVER_PORT'] = '107'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--theme.base=light",
        "--theme.primaryColor=#1DA1F2",
        "--theme.font=sans serif"
    ]
    process = subprocess.Popen(cmd)
    time.sleep(3)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()


if __name__ == "__main__":
    start_streamlit()
