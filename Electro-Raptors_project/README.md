# ESP32 Signal Pipeline

Batch pipeline for raw CSV sensor data using NumPy and the Python standard library:

`CSV -> validation -> preprocessing -> filtering -> windowing -> features -> FFT -> ML-ready CSV`

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[test]"
```

## Run tests

```powershell
py -m pytest
```

## Run the pipeline

The input CSV must contain `timestamp` and one or more numeric sensor columns. An optional `label` column is aggregated per window.

```powershell
signal-pipeline data\raw.csv output\features.csv --sensors accel_x accel_y accel_z --sampling-rate 100 --window-seconds 2 --overlap 0.5 --lowcut-hz 0.5 --highcut-hz 20
```

The first version is intentionally batch-oriented. FastAPI is not required. Add an API only when another application or the ESP32 needs to send data to this pipeline over a network.

## Train a baseline model

The feature CSV must contain at least two distinct values in its optional `label` column:

```powershell
signal-train output\features.csv output\model.json
```

This creates a NumPy-only nearest-centroid model. The demo CSV contains `idle` and `active`; for real work, use labeled recordings such as `idle`, `walking`, and `running`.

## Run the AI stage

The signal pipeline and the AI project are connected by the ML-ready feature CSV. From `Electro-Raptors/Vibration-Predictive-Maintenance`, install the AI dependencies and run:

```powershell
py -m pip install -r requirements.txt
py -m ai.src.cli ..\..\output\features.csv ai\models\anomaly_detector.joblib --classifier-model ai\models\fault_classifier.joblib
```

The command loads the feature windows, validates the dataset, runs anomaly detection, optionally classifies faults, and reports the degradation trend and early-warning level. Train and save those AI artifacts with the `AnomalyDetector` and `FaultClassifier` classes in `ai/src`.

## Read directly from ESP32

The ESP32 can stream either `timestamp,accel_x,accel_y,accel_z` or just `accel_x,accel_y,accel_z` at the configured sampling rate. No manual CSV step is required:

```powershell
signal-serial --port COM5 --baudrate 115200 --sensors accel_x accel_y accel_z --sampling-rate 100 --duration-seconds 10 --model output\model.json --raw-output output\esp32_capture.csv
```

The command captures serial data automatically, extracts windows, and prints model predictions. `--raw-output` is optional and only keeps an audit copy.

## ESP32 firmware and piezo input

Your 27 mm piezo disc is the sensor. The buzzer, LED, and motor are outputs; they do not provide sensor readings. The device-side example is [esp32/micropython/main.py](esp32/micropython/main.py), which reads the piezo through ESP32 ADC GPIO34 at 100 Hz and prints one `piezo` value per line.

This project assumes MicroPython on the ESP32, not Arduino. Upload `main.py` with Thonny, `mpremote`, or another MicroPython tool. Set the serial speed to `115200` and close the MicroPython terminal before starting `signal-serial`.

Do not connect the piezo directly to an ADC pin. A piezo can generate voltage spikes. Use a biased, protected input: bias the ADC around 1.65 V with two 100k resistors, use the 10 microfarad capacitor to stabilize that bias, couple the piezo through the 10 nanofarad capacitor, and clamp/protect the ADC with suitable series resistance and diodes. Confirm the resulting voltage never leaves the ESP32 ADC input range before sampling.

The 1N4148 diodes and resistors should be used for protection, not as a substitute for checking the circuit with a multimeter. Drive the motor through a transistor or motor driver; do not drive it directly from an ESP32 GPIO.

Run the direct reader with one sensor name:

```powershell
signal-serial --port COM7 --baudrate 115200 --sensors piezo --sampling-rate 100 --duration-seconds 10 --model output\model.json --raw-output output\esp32_capture.csv
```

## Predict and evaluate

Generate features for a new recording using the same sensor and window settings, then run:

```powershell
signal-predict output\features.csv output\model.json output\predictions.csv
```

When the feature CSV includes `label`, the command also prints accuracy. For an honest evaluation, use a separate recording that was not used for training.

## Software-only piezo test

Until the ESP32 is connected, generate a labeled piezo fixture and run the same pipeline without entering sensor values manually:

```powershell
signal-demo data\piezo_demo.csv
signal-pipeline data\piezo_demo.csv output\piezo_features.csv --sensors piezo --sampling-rate 100 --window-seconds 2 --overlap 0.5
signal-train output\piezo_features.csv output\piezo_model.json
signal-predict output\piezo_features.csv output\piezo_model.json output\piezo_predictions.csv
```

This fixture validates the software only; it is not a substitute for real piezo readings.

## Dashboard and validation

The dashboard reads the latest generated feature or prediction CSV automatically. It does not accept manual sensor values or manually entered health states.

From the project root, start it with:

```powershell
\.venv\Scripts\python.exe dashboard\server.py
```

Open `http://127.0.0.1:8765`. The dashboard displays the latest backend prediction, vibration RMS, dominant frequency, signal trend, dataset quality checks, and prediction accuracy. It refreshes automatically as the backend output CSV changes.
