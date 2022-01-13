import subprocess
import time

while True:
    print("[INFO] Start training new model")
    subprocess.Popen(["python","vasp.py"], close_fds=True)
    time.sleep(3000)
