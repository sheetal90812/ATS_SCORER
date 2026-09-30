from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://localhost:8000"


def _backend_url() -> str:
    try:
        return st.secrets["backend"]["url"]
    except (KeyError, FileNotFoundError):
        return DEFAULT_BACKEND_URL


def _auth_headers(access_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}

def health_check() -> Dict[str, Any]:
    response = requests.get(f"{_backend_url()}/api/v1/health", timeout=10)
    response.raise_for_status()
    return response.json()

def analyze_resume(
    resume_file,
    access_token: str,
    job_description: str = "",
) -> Dict[str, Any]:

    print("DEBUG: analyze_resume() CALLED")

    backend_url = _backend_url()
    health_url = f"{backend_url}/api/v1/health"

    # Wake the Render backend and wait until it is ready.
    for attempt in range(12):
        try:
            health_response = requests.get(
                health_url,
                timeout=10,
            )

            if health_response.ok:
                print(
                    f"DEBUG: backend healthy on attempt {attempt + 1}"
                )
                break

            print(
                f"DEBUG: backend wake attempt {attempt + 1} "
                f"returned {health_response.status_code}"
            )

        except requests.RequestException as exc:
            print(
                f"DEBUG: backend wake attempt {attempt + 1} "
                f"failed: {exc}"
            )

        if attempt == 11:
            raise RuntimeError(
                "Backend is taking too long to start. Please try again."
            )

    files = {
        "resume": (
            resume_file.name,
            resume_file.getvalue(),
            resume_file.type,
        ),
    }

    data = {"job_description": job_description}

    print("DEBUG ANALYZE BACKEND URL:", backend_url)

    print(
        "DEBUG AUTH TOKEN:",
        "present" if access_token else "MISSING",
        "length=",
        len(access_token) if access_token else 0,
    )

    response = requests.post(
        f"{backend_url}/api/v1/analyze-resume",
        files=files,
        data=data,
        headers=_auth_headers(access_token),
        timeout=180,
    )

    response.raise_for_status()
    return response.json()