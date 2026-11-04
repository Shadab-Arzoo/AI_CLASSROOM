"""
ExamGuard - Main Application Launcher
Starts the backend API and opens the Invigilator Dashboard in your default browser.
"""

import sys
import time
import webbrowser
import threading
import uvicorn
from pathlib import Path

def open_browser(port: int):
    """Wait for server to start, then launch browser."""
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[+] Opening ExamGuard Dashboard at: {url}")
    webbrowser.open(url)

def main():
    port = 8000
    host = "0.0.0.0"

    print("=" * 65)
    print("        EXAMGUARD: AI-BASED EXAM MONITORING SYSTEM")
    print("=" * 65)
    print(f" * Server Address: http://{host}:{port}")
    print(" * Invigilator Dashboard: Interactive Web UI")
    print(" * AI Engine: Ultralytics YOLO & ByteTrack")
    print(" * Database: SQLite (Persistent Incident Records)")
    print("=" * 65)

    # Launch browser in a background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start Uvicorn Server
    try:
        uvicorn.run("src.server:app", host=host, port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[!] ExamGuard Server Stopped Gracefully.")

if __name__ == "__main__":
    main()

