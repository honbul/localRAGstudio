import json
import os
import sys
import urllib.request
import urllib.error

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception as exc:
    print(f"Failed to load Tkinter: {exc}")
    sys.exit(1)


def pick_folder() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder or None


def post_path(path: str, endpoint: str) -> None:
    payload = json.dumps({"path": path}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            print(body)
    except urllib.error.URLError as exc:
        print(f"Failed to send path: {exc}")
        sys.exit(1)


def main() -> None:
    endpoint = os.environ.get("FOLDER_HELPER_ENDPOINT", "http://localhost:8000/api/folder-path")
    folder = pick_folder()
    if not folder:
        print("No folder selected")
        return
    post_path(os.path.abspath(folder), endpoint)


if __name__ == "__main__":
    main()
