"""
Unified Application Entry Point for AI Resume Screening & ATS Score Predictor.
Allows starting FastAPI backend, Streamlit frontend, or both concurrently.
"""

import argparse
import os
import subprocess
import sys
import time


def check_and_prepare_dirs():
    """Ensures required working directories exist."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "sample_resumes"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "sample_job_descriptions"), exist_ok=True)


def start_backend(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Starts the FastAPI server with Uvicorn."""
    print(f"[*] Starting FastAPI backend at http://{host}:{port} ...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.extend(["--reload", "--reload-dir", "backend"])
    return subprocess.Popen(cmd)



def start_frontend(port: int = 8501):
    """Starts the Streamlit dashboard."""
    print(f"[*] Starting Streamlit frontend at http://localhost:{port} ...")
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dashboard.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_path,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & ATS Score Predictor Launcher"
    )
    parser.add_argument(
        "--backend", action="store_true", help="Launch FastAPI backend only"
    )
    parser.add_argument(
        "--frontend", action="store_true", help="Launch Streamlit frontend only"
    )
    parser.add_argument(
        "--both", action="store_true", help="Launch both backend and frontend concurrently"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run NLP model comparison benchmark"
    )
    parser.add_argument(
        "--generate-data", action="store_true", help="Generate synthetic resumes and job descriptions"
    )
    parser.add_argument(
        "--port-backend", type=int, default=8000, help="Backend port (default: 8000)"
    )
    parser.add_argument(
        "--port-frontend", type=int, default=8501, help="Frontend port (default: 8501)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable live code reloading for development"
    )

    args = parser.parse_args()
    check_and_prepare_dirs()

    if args.generate_data:
        from data.generate_data import generate_all
        generate_all()
        return

    if args.benchmark:
        from models.compare_models import run_benchmark
        run_benchmark()
        return

    if args.backend:
        proc = start_backend(port=args.port_backend, reload=args.reload)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    if args.frontend:
        proc = start_frontend(port=args.port_frontend)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    # Default action: launch both if none or --both specified
    print("=" * 70)
    print("  AI Resume Screening & ATS Score Predictor Platform")
    print("=" * 70)
    print("Launching FastAPI Backend on http://localhost:8000 ...")
    backend_proc = start_backend(port=args.port_backend, reload=args.reload)
    time.sleep(2)


    print("Launching Streamlit Frontend on http://localhost:8501 ...")
    frontend_proc = start_frontend(port=args.port_frontend)

    print("\n[+] Both services are operational!")
    print("    - Streamlit Dashboard: http://localhost:8501")
    print("    - FastAPI Swagger Docs: http://localhost:8000/docs")
    print("    - FastAPI ReDoc Docs:   http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop both servers gracefully.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Shutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("[+] Servers stopped cleanly.")


if __name__ == "__main__":
    main()
