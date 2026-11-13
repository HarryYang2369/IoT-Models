# Running pipeline v4 on the Windows laptop (RTX 4060)

The Mac and the laptop split the 70 jobs: the **Mac runs folds 0–4** (already running) and the **laptop runs
folds 5–9**. At the end, the laptop's result files are copied into the Mac's `results/jobs/` folder and the report
is rebuilt. Every job is an independent file, so nothing is run twice and nothing is lost if you stop and restart.

TensorFlow can only use an NVIDIA GPU on Windows **through WSL2** (Ubuntu running inside Windows). Native Windows
TensorFlow has been CPU-only since version 2.10. This guide uses WSL2. A CPU-only fallback is at the end.

Allow about 30–45 minutes for the first-time setup (steps 1–6). It's mostly downloads.

---

## Step 1: update the NVIDIA driver (Windows)

1. Open the **NVIDIA App** (or GeForce Experience) → Drivers → install the latest Game Ready or Studio driver.
   Alternatively, download it from nvidia.com/drivers (GeForce RTX 40 Series (Notebooks) → RTX 4060 Laptop GPU).
2. Restart Windows.

Do **not** install any NVIDIA or CUDA driver inside Ubuntu later. WSL uses the Windows driver automatically.

## Step 2: install WSL2 with Ubuntu (Windows)

1. Right-click Start → **Terminal (Admin)** (or PowerShell as administrator).
2. Run:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. Restart Windows when asked. Ubuntu opens and asks for a **new Linux username and password** (it can be anything;
   you'll need the password for `sudo`).
4. Back in an admin PowerShell, make sure everything is current and that Ubuntu runs as WSL **2**:
   ```powershell
   wsl --update
   wsl -l -v
   ```
   The `VERSION` column must show `2`.

From here on, every command goes in the **Ubuntu** window (Start menu → Ubuntu) unless it says otherwise.

## Step 3: check that Ubuntu sees the GPU

```bash
nvidia-smi
```

You should see a table naming **NVIDIA GeForce RTX 4060 Laptop GPU**. If you get "command not found" or an error, go
back to step 1 (driver), then run `wsl --update` in PowerShell, and run `wsl --shutdown` before reopening Ubuntu.

## Step 4: install the system tools

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip zip unzip
python3 --version        # should print Python 3.12.x
```

## Step 5: copy the code and data to the laptop

**On the Mac**, make one zip with exactly what the laptop needs: the data, the model library and pipeline v4.
Results and the cache are left out; the laptop builds its own.

```bash
cd "/Volumes/Macintosh HD2/IoT-Models"
zip -rq ~/Desktop/pdiot_v4_laptop.zip data model_harry/harry model_harry/pipeline_v4 \
    -x "*/results/*" "*/results_smoke/*" "*/cache/*" "*__pycache__*" "*.DS_Store"
ls -lh ~/Desktop/pdiot_v4_laptop.zip
```

Move `pdiot_v4_laptop.zip` to the laptop (USB stick formatted exFAT, OneDrive, Google Drive, AirDrop to a phone, etc.)
and put it in the Windows **Downloads** folder.

**In Ubuntu**, copy it into the Linux home folder and unzip it. Work in `~`, not in `/mnt/c/...`: the Windows drive
is much slower from WSL.

```bash
cd ~
cp /mnt/c/Users/<WindowsUser>/Downloads/pdiot_v4_laptop.zip .    # replace <WindowsUser> with your Windows user name
mkdir -p IoT-Models && unzip -q pdiot_v4_laptop.zip -d IoT-Models
ls IoT-Models IoT-Models/model_harry
```

You should see `data` in `IoT-Models/`, and `harry` and `pipeline_v4` in `IoT-Models/model_harry/`. The folder layout must
stay exactly like this, because the code finds the data at `IoT-Models/data/`.

(Alternative: if you commit and push `model_harry/pipeline_v4` to your IoT-Models GitHub repo, you can instead run
`git clone https://github.com/HarryYang2369/IoT-Models.git` in `~`. The data is already in the repo.)

## Step 6: create the Python environment with GPU TensorFlow

```bash
cd ~/IoT-Models
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r model_harry/pipeline_v4/requirements-gpu.txt      # ~3 GB of downloads (CUDA libraries included)
```

Check that TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Expected output: `2.20.0 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. If the list is empty
(`[]`), see *Troubleshooting*.

`run.sh` finds this environment automatically (`IoT-Models/.venv`). You don't need to activate it again later.

## Step 7: smoke test (about 3–5 minutes)

```bash
cd ~/IoT-Models/model_harry/pipeline_v4
./run.sh --smoke --gpu --workers 2
```

- The first time, it builds the data cache (`cached har ... / cached social ... / cached real data: 88 recordings`).
- Then 7 lines ending `ok in 0.x min`, one per candidate, and `summary -> .../results_smoke/report.md`.
- Confirm it really used the GPU:
  ```bash
  tail -2 results_smoke/report.md        # must say "Jobs run on: GPU on Linux x86_64 (7)"
  rm -rf results_smoke models_smoke      # delete the test output (it is not a result)
  ```

If a line says `failed`, open its log: `cat results_smoke/jobs/<job id>.log`.

## Step 8: before the real run (Windows settings)

- Plug the charger in. Battery mode throttles the GPU.
- Settings → System → Power → **Power mode: Best performance**; **Screen and sleep: Never** (when plugged in).
- Control Panel → Power Options → *Choose what closing the lid does* → **Do nothing** (plugged in), or keep the lid open.
- Keep the Ubuntu window open. Closing it stops the run.
- If the laptop has a vendor app (Armoury Crate, Lenovo Vantage, Omen Gaming Hub, etc.), choose its **Performance/Turbo** mode.

## Step 9: the real run, folds 5–9 on the GPU

```bash
cd ~/IoT-Models/model_harry/pipeline_v4
./run.sh --folds 5,6,7,8,9 --gpu --workers 4
```

- 35 jobs (7 candidates × 5 folds). Progress lines look like
  `[3/35] cnn_w75_jan_s0_f5 ok in 6.2 min · elapsed 0.2 h · est. left 1.4 h`.
- On the Mac, each CNN job takes about 16 minutes. The first few laptop lines tell you how much faster the laptop is.
- Optional: monitor the GPU in a second Ubuntu window with `watch -n 5 nvidia-smi`.
  - If **GPU-Util** stays low (below ~50 %) and plenty of memory is free, stop with `Ctrl+C` and restart with
    `--workers 6`. Finished jobs are kept; only the jobs that were running restart.
  - If a job fails with an out-of-memory error, restart with `--workers 3`.
- Stopped by accident (sleep, closed window, Ctrl+C)? Run the same command again and it continues.

**Only ever run folds 5–9 on the laptop.** The Mac is running folds 0–4. If the laptop finishes first and you want it
to help with the Mac's remaining folds, ask Claude on the Mac to re-split them, so no job runs on both.

## Step 10: send the results back to the Mac

**In Ubuntu**, when the run prints `summary -> ...`:

```bash
cd ~/IoT-Models/model_harry/pipeline_v4/results
zip -rq /mnt/c/Users/<WindowsUser>/Desktop/v4_laptop_jobs.zip jobs
```

Move `v4_laptop_jobs.zip` (on the Windows Desktop) to the Mac, for example into Downloads. Then, **on the Mac**:

```bash
cd "/Volumes/Macintosh HD2/IoT-Models/model_harry/pipeline_v4/results"
unzip -n ~/Downloads/v4_laptop_jobs.zip        # -n: never overwrite a file that is already there
cd .. && ./run.sh --summary-only
```

Open `results/report.md`. Its last line lists where the jobs ran, for example
`Jobs run on: CPU on Darwin arm64 (35), GPU on Linux x86_64 (35)`. Each candidate's `folds` column must be **10**.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` not found / no GPU in Ubuntu | Update the Windows NVIDIA driver (step 1), run `wsl --update` and then `wsl --shutdown` in PowerShell, reopen Ubuntu. |
| TensorFlow prints `[]` for the GPU | Make sure the venv is active (`source ~/IoT-Models/.venv/bin/activate`) and that you installed `requirements-gpu.txt` (it contains `tensorflow[and-cuda]`). Run `pip install --force-reinstall "tensorflow[and-cuda]==2.20.0"`. |
| `--gpu was given but TensorFlow sees no GPU` in a job log | Same as above. The pipeline refuses to fall back silently to the CPU. |
| Jobs killed, or `Killed` with no error | WSL ran out of memory. Use `--workers 3`, or create `C:\Users\<WindowsUser>\.wslconfig` with the two lines `[wsl2]` and `memory=12GB`, then run `wsl --shutdown` and reopen Ubuntu. |
| `No Python with tensorflow ... found` | Step 6 wasn't completed in `~/IoT-Models/.venv`, or you ran `run.sh` from another folder layout. |
| `FileNotFoundError ... data/fold_assignments` | The folder layout is wrong: `IoT-Models/data/` and `IoT-Models/model_harry/pipeline_v4/` must both exist (step 5). |
| Very slow (hours per job) | Check `nvidia-smi` shows the python processes, the charger is plugged in, and Power mode is Best performance. |

## CPU-only alternative (native Windows, no WSL, no GPU)

Slower than the GPU, but simple. Install Python 3.12 from python.org (tick "Add python.exe to PATH"), unzip the zip from
step 5 to `C:\IoT-Models`, then in PowerShell:

```powershell
cd C:\IoT-Models
py -3.12 -m venv .venv
.venv\Scripts\pip install tensorflow==2.20.0 numpy pandas scikit-learn scipy matplotlib pyyaml
cd model_harry\pipeline_v4
..\..\.venv\Scripts\python run_v4.py --smoke --workers 2       # test
..\..\.venv\Scripts\python run_v4.py --folds 5,6,7,8,9 --workers 4
```

Copy `model_harry\pipeline_v4\results\jobs` back to the Mac as in step 10.
