import subprocess
import time
import urllib


def run_subprocess(cmd: str, url: str | None = None) -> subprocess.Popen:
    proc = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if url is not None:
        if not wait_healthy(url, timeout=90):
            out = proc.stdout.read().decode(errors="replace")[-2000:] if proc.stdout else ""
            proc.terminate()
            raise RuntimeError(f"There is an issue accessing the port-mapping or url at {url}! Please verify your configuration and try again.")
    return proc


def wait_healthy(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False
