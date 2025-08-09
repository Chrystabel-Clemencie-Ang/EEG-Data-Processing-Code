import serial
import struct
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import tkinter as tk
from tkinter import simpledialog
import os
import joblib

import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter-compatible backend for Matplotlib

# Load model
try:
    model = joblib.load("svm_eeg_model.pkl")
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    model = None



# --- Constants ---
LABELS = ['forward', 'backward', 'left', 'right', 'stop']
DATASET_FILE = 'eeg_raw_dataset.csv'
MAX_LEN = 5000  # Cap memory usage

# --- CSV Setup ---
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"raw_{i}" for i in range(1024)] + ['label']
        writer.writerow(header)

# --- Serial Port ---
try:
    ser = serial.Serial('COM4', 57600, timeout=1)
except Exception as e:
    print(f"[ERROR] Cannot open serial port: {e}")
    exit(1)

# --- Data Storage ---
attention_vals = []
meditation_vals = []
raw_vals = []
eeg_bands = {'delta': 0, 'theta': 0, 'lowAlpha': 0, 'highAlpha': 0,
             'lowBeta': 0, 'highBeta': 0, 'lowGamma': 0, 'midGamma': 0}

data_lock = threading.Lock()

# --- Decode band types ---
band_map = {
    0x83: 'delta',
    0x84: 'theta',
    0x85: 'lowAlpha',
    0x86: 'highAlpha',
    0x87: 'lowBeta',
    0x88: 'highBeta',
    0x89: 'lowGamma',
    0x8A: 'midGamma'
}

prediction_label = "Waiting..."
prediction_lock = threading.Lock()

def prediction_loop():
    global prediction_label
    while True:
        with data_lock:
            if len(raw_vals) >= 1024:
                segment = np.array(raw_vals[-1024:], dtype=np.float32).reshape(1, -1)
            else:
                segment = None
        if segment is not None and model is not None:
            try:
                pred = model.predict(segment)[0]
                with prediction_lock:
                    prediction_label = str(pred)
                print(f"[PREDICTION] {prediction_label}")
            except Exception as e:
                print("[Prediction error]", e)
        time.sleep(2)  # update setiap 2 detik

# Start prediction thread setelah serial thread
threading.Thread(target=prediction_loop, daemon=True).start()


# --- Packet Parser ---
def parse_packet(data):
    i = 0
    result = {}
    while i < len(data) - 3:
        if data[i] == 0xAA and data[i+1] == 0xAA:
            pkt_len = data[i+2]
            if i + 3 + pkt_len <= len(data):
                payload = data[i+3:i+3+pkt_len]
                j = 0
                while j < len(payload):
                    code = payload[j]
                    if code == 0x02 and j + 2 < len(payload):  # raw EEG
                        val = struct.unpack('>h', bytes(payload[j+1:j+3]))[0]
                        result['raw'] = val
                        j += 3
                    elif code == 0x04:  # attention
                        result['attention'] = payload[j+1]
                        j += 2
                    elif code == 0x05:  # meditation
                        result['meditation'] = payload[j+1]
                        j += 2
                    elif code in band_map and j+3 < len(payload):
                        val = int.from_bytes(payload[j+1:j+4], byteorder='big')
                        result[band_map[code]] = val
                        j += 4
                    else:
                        j += 1
                i += 3 + pkt_len
            else:
                break
        else:
            i += 1
    return result

# --- Serial Reader Thread ---
def serial_reader():
    while True:
        try:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                values = parse_packet(data)
                with data_lock:
                    if 'attention' in values:
                        attention_vals.append(values['attention'])
                        if len(attention_vals) > MAX_LEN:
                            attention_vals.pop(0)
                    if 'meditation' in values:
                        meditation_vals.append(values['meditation'])
                        if len(meditation_vals) > MAX_LEN:
                            meditation_vals.pop(0)
                    if 'raw' in values:
                        raw_vals.append(values['raw'])
                        if len(raw_vals) > MAX_LEN:
                            raw_vals.pop(0)
                    for k in eeg_bands:
                        if k in values:
                            eeg_bands[k] = values[k]
        except Exception as e:
            print("[Serial Error]", e)
        time.sleep(0.01)

# --- Plotting Function ---
def animate(frame):
    with data_lock, prediction_lock:
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        axs[3].clear()

        axs[0].plot(attention_vals[-100:], label='Attention', color='blue')
        axs[0].plot(meditation_vals[-100:], label='Meditation', color='green')
        axs[0].legend()
        axs[0].set_title(f'Attention & Meditation | Prediction: {prediction_label}')
        axs[0].set_ylim(0, 100)

        axs[1].plot(raw_vals[-256:], color='purple')
        axs[1].set_title('Raw EEG')

        if len(raw_vals) >= 256:
            y = np.array(raw_vals[-256:])
            fft = np.abs(np.fft.rfft(y))
            freq = np.fft.rfftfreq(len(y), d=1/512)
            axs[2].plot(freq, fft, color='orange')
            axs[2].set_title('EEG Spectrum')
            axs[2].set_xlim(0, 60)

        axs[3].bar(eeg_bands.keys(), eeg_bands.values(), color='teal')
        axs[3].set_title('EEG Power Bands')

# --- Data Collection ---
def record_eeg_for_label(label, duration=2, sampling_rate=512):
    sample_count = duration * sampling_rate
    print(f"[INFO] Collecting {sample_count} samples for label: {label}")

    segment = []
    while len(segment) < sample_count:
        with data_lock:
            if raw_vals:
                segment.append(raw_vals[-1])
        time.sleep(1.0 / sampling_rate)

    with open(DATASET_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = segment[:sample_count] + [label]
        writer.writerow(row)
    print(f"[INFO] Saved {sample_count} samples for label '{label}'")
#
# # --- GUI (main thread) ---
# def start_label_gui():
#     def on_label_click(label):
#         threading.Thread(target=record_eeg_for_label, args=(label,), daemon=True).start()
#
#     win = tk.Tk()
#     win.title("EEG Labeling")
#     for i, label in enumerate(LABELS):
#         btn = tk.Button(win, text=label.capitalize(), command=lambda l=label: on_label_click(l), width=20)
#         btn.grid(row=i, column=0, pady=5, padx=10)
#     return win

# --- Start Everything ---
threading.Thread(target=serial_reader, daemon=True).start()

fig, axs = plt.subplots(4, 1, figsize=(10, 10))
fig.tight_layout(pad=3.0)
ani = FuncAnimation(fig, animate, interval=500)

# Tkinter GUI must be run in main thread, so combine with matplotlib
# gui = start_label_gui()

# Integrate both GUIs (Matplotlib + Tkinter)
def run_gui_and_plot():
    plt.show()
    # gui.mainloop()

# Run both GUIs
run_gui_and_plot()
