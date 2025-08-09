import argparse
import pandas as pd
import numpy as np
import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import matplotlib

matplotlib.use('TkAgg')  # Menggunakan backend TkAgg untuk Matplotlib

# --- Konstanta & Konfigurasi ---
number_of_bytes = 512  # Jumlah byte yang dibaca dari port serial setiap kali
RAW_DATA_FILE = "../model/maju/raw_data.csv"  # Lokasi file CSV untuk data mentah
BAND_DATA_FILE = "../model/maju/band_data.csv"  # Lokasi file CSV untuk data band
MAX_SAMPLES = 500  # Batas jumlah sampel data band yang akan diambil

# --- Buat folder jika belum ada ---
os.makedirs("../model/maju", exist_ok=True)

# --- Kode byte dari EEG (NeuroSky Mindwave) ---
CONNECT = 0xc0
SYNC = 0xaa
RAW_VALUE = 0x80
ASIC_EEG_POWER = 0x83


# --- Parser Data Otak ---
class BrainWaveDataParser:
    def __init__(self):
        self.parse_data = self._parse_data()
        self.raw_data_buffer = []  # Buffer untuk menyimpan nilai mentah EEG sementara
        self.band_data_buffer = []  # Buffer untuk menyimpan data kekuatan band EEG sementara
        next(self.parse_data)  # Memulai generator

    def get_data(self, data):
        for c in data:
            self.parse_data.send(c)

    def _parse_data(self):
        while True:
            byte = yield
            if byte == SYNC:
                byte = yield
                if byte == SYNC:
                    packet_length = yield
                    packet_code = yield
                    if packet_code == CONNECT:
                        continue

                    left = packet_length - 2
                    while left > 0:
                        if packet_code == RAW_VALUE:
                            low = yield
                            high = yield
                            val = (high << 8) | low
                            if val > 32768:
                                val -= 65536
                            self.raw_data_buffer.append(val)
                            left -= 2
                        elif packet_code == ASIC_EEG_POWER:
                            vector_length = yield
                            vector = []
                            for _ in range(8):
                                low = yield
                                mid = yield
                                high = yield
                                value = (high << 16) | (mid << 8) | low
                                vector.append(value)
                            left -= vector_length

                            total = sum(vector)
                            if total > 0:
                                norm = [v / total for v in vector]
                                self.band_data_buffer.append({
                                    "timestamp": time.time(),
                                    "delta": norm[0],
                                    "theta": norm[1],
                                    "low-alpha": norm[2],
                                    "high-alpha": norm[3],
                                    "low-beta": norm[4],
                                    "high-beta": norm[5],
                                    "low-gamma": norm[6],
                                    "mid-gamma": norm[7],
                                })
                        if left > 0:
                            packet_code = yield


# --- Fungsi koneksi EEG ---
def mindwave_connect(port, baud_rate=57600, timeout=1):
    try:
        with serial.Serial(port=port, baudrate=baud_rate, timeout=timeout) as ser:
            return ser.read(number_of_bytes)
    except serial.SerialException as e:
        print(f"[ERROR] Port serial '{port}' tidak bisa dibuka atau diakses: {e}")
        return b''
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat membaca serial dari '{port}': {e}")
        return b''


# --- Simpan Data ke CSV ---
def save_raw_data(data_to_save):
    if data_to_save:
        with open(RAW_DATA_FILE, "a") as f:
            for v in data_to_save:
                f.write(f"{time.time()},{v}\n")


def save_band_data(data_to_save):
    if data_to_save:
        df = pd.DataFrame(data_to_save)
        df.to_csv(BAND_DATA_FILE, mode='a', index=False, header=not os.path.exists(BAND_DATA_FILE))


# --- Grafik Real-time ---
def update_plot(frame):
    global samples_collected
    global ani  # Diperlukan untuk menghentikan animasi
    global fig  # Diperlukan untuk menutup figure

    data = mindwave_connect(port_address)
    brain_data.get_data(data)

    current_raw_data_for_save = list(brain_data.raw_data_buffer)
    current_band_data_for_save = list(brain_data.band_data_buffer)

    brain_data.raw_data_buffer = []
    brain_data.band_data_buffer = []

    save_raw_data(current_raw_data_for_save)
    save_band_data(current_band_data_for_save)

    samples_collected += len(current_band_data_for_save)

    global plot_band_history
    plot_band_history.extend(current_band_data_for_save)
    plot_band_history = plot_band_history[-50:]  # Batasi untuk plot 50 data terakhir

    if plot_band_history:
        plt.cla()
        df = pd.DataFrame(plot_band_history)

        colors = ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'cyan', 'magenta']

        for i, band in enumerate(df.columns[1:]):
            plt.plot(df.index, df[band], label=band, color=colors[i % len(colors)])

        plt.title(f"Real-time EEG Band Power (Sampel: {samples_collected}/{MAX_SAMPLES})")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Amplitude")
        plt.legend(loc='upper right', fontsize='small')
        plt.ylim(0, 1)
        plt.grid(True)

    print(
        f"[INFO] Terkumpul dan disimpan: {len(current_band_data_for_save)} data band. {len(current_raw_data_for_save)} data mentah. Total sampel band: {samples_collected}/{MAX_SAMPLES}")

    if samples_collected >= MAX_SAMPLES:
        print(f"[INFO] Batas {MAX_SAMPLES} sampel tercapai. Menghentikan pengambilan data.")
        ani.event_source.stop()  # Menghentikan animasi
        plt.close(fig)  # Menutup jendela plot
        # os._exit(0) # Hanya uncomment ini jika program tidak otomatis keluar setelah plt.close(fig)


# --- Argument Parser & Inisialisasi ---
parser = argparse.ArgumentParser(description="Program untuk membaca dan menampilkan data EEG dari NeuroSky Mindwave.")
parser.add_argument('--address', type=str, default='COM4',
                    help="Serial port EEG (contoh: COM4 di Windows, /dev/ttyUSB0 di Linux). Default: COM4")
args = parser.parse_args()
port_address = args.address

brain_data = BrainWaveDataParser()

plot_band_history = []
samples_collected = 0

print(f"[INFO] Mencoba koneksi ke Mindwave di {port_address}...")

initial_data_attempts = 0
initial_data_success = False
while initial_data_attempts < 5 and not initial_data_success:
    initial_data = mindwave_connect(port_address)
    if initial_data:
        brain_data.get_data(initial_data)
        if brain_data.band_data_buffer or brain_data.raw_data_buffer:
            initial_data_success = True
    else:
        print(f"[INFO] Percobaan koneksi awal ke-{initial_data_attempts + 1} gagal. Mencoba lagi...")
        time.sleep(0.5)
    initial_data_attempts += 1

if not initial_data_success:
    print(
        "[ERROR] Gagal mendapatkan data awal setelah beberapa percobaan. Pastikan perangkat terhubung dan port serial benar.")
    exit()

plot_band_history.extend(brain_data.band_data_buffer)
samples_collected += len(brain_data.band_data_buffer)

print(f"[INFO] Koneksi berhasil ke {port_address}. Menampilkan grafik real-time dan mulai merekam data...")

# --- Tampilkan grafik ---
fig, ax = plt.subplots(figsize=(12, 6))
# Tambahkan cache_frame_data=False untuk menghindari UserWarning dari Matplotlib
ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)
plt.show()

print("[INFO] Grafik ditutup. Program selesai.")
print(f"[✅ SELESAI] Data EEG berhasil dikumpulkan dan disimpan di:\n- {RAW_DATA_FILE}\n- {BAND_DATA_FILE}")