import subprocess
import time
import requests
import os

# =============================
# CONFIG
# =============================

# 1. List of dataset IDs to run through
DATASETS = [760]   # <- EDIT THESE

# 2. Telegram credentials
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# =============================
# FUNCTIONS
# =============================

def send_telegram(message: str):
    """Send a Telegram message to your chat."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send telegram message: {e}")


def run_command(dataset_id: int):
    """Runs the nnUNet training command for a given dataset."""
    num_gpus = 4
    if dataset_id == 670 or dataset_id == 673:
        num_gpus = 1
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        "2d",
        "0",
        "-num_gpus",
        f"{num_gpus}"
    ]

    # Notify start
    send_telegram(f"Starting training for DATASET_ID {dataset_id}")

    # Run command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # You can check return codes if you want
    if result.returncode != 0:
        send_telegram(f"❌ Training FAILED for DATASET_ID {dataset_id} (code {result.returncode})")
    else:
        send_telegram(f"✅ Finished training for DATASET_ID {dataset_id}")


# =============================
# MAIN EXECUTION
# =============================

if __name__ == "__main__":
    start_time = time.time()

    for ds in DATASETS:
        run_command(ds)

    total = time.time() - start_time
    send_telegram(f"🎉 PROGRAM DONE — total time: {total/60:.2f} minutes")
    print("PROGRAM DONE")
