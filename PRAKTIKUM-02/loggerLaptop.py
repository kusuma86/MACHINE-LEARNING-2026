"""logger.py -- jalankan dengan Python biasa di laptop (bukan MicroPython).
Instal dependensi: pip install pyserial
Tutup koneksi Thonny ke COM port sebelum menjalankan program.
"""
import csv
import time
from pathlib import Path

import serial

PORT = 'COM14'             # Sesuaikan dengan Device Manager / ls /dev/ttyUSB*
BAUD = 115200
FS = 200                   # Harus sama dengan main.py
DURATION_SECONDS = 30      # 30 detik = 6000 sampel
TOTAL_SAMPLES = FS * DURATION_SECONDS
DATA_DIR = Path('data')


def read_line(ser):
    return ser.readline().decode('utf-8', errors='replace').strip()


def handshake(ser):
    # Sebagian board reset ketika port serial dibuka.
    time.sleep(2)
    ser.reset_input_buffer()
    for _ in range(4):
        ser.write(b'PING\n')
        ser.flush()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            line = read_line(ser)
            if line == 'PONG':
                return
    raise RuntimeError(
        'ESP32 tidak membalas PING. Pastikan main.py berjalan dan Thonny tidak memakai port.'
    )


def capture(ser, output_file):
    # Buang sisa pesan dari rekaman sebelumnya.
    ser.reset_input_buffer()
    ser.write('START,{}\n'.format(TOTAL_SAMPLES).encode('ascii'))
    ser.flush()

    # Tunggu konfirmasi START sebelum membuat file CSV.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        line = read_line(ser)
        if line == 'BEGIN,{}'.format(TOTAL_SAMPLES):
            break
        if line.startswith('ERROR,'):
            raise RuntimeError('ESP32: ' + line)
    else:
        raise TimeoutError('ESP32 tidak mengonfirmasi BEGIN.')

    rows = 0
    expected_seq = 0
    errors = 0
    ended = False
    last_rx = time.monotonic()
    last_time_us = None

    with output_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 't_us', 'ax_raw', 'ay_raw', 'az_raw'])
        while True:
            line = read_line(ser)
            if not line:
                if time.monotonic() - last_rx > 5:
                    raise TimeoutError('Tidak ada data serial selama 5 detik.')
                continue
            last_rx = time.monotonic()
            if line == 'END,{}'.format(TOTAL_SAMPLES):
                ended = True
                break
            if line.startswith('ERROR,'):
                raise RuntimeError('ESP32: ' + line)
            if not line.startswith('D,'):
                continue  # Abaikan pesan non-data, seperti output boot.
            try:
                tag, seq, t_us, ax, ay, az = line.split(',')
                seq, t_us, ax, ay, az = map(int, (seq, t_us, ax, ay, az))
            except ValueError:
                errors += 1
                continue
            if seq != expected_seq:
                errors += abs(seq - expected_seq)
            expected_seq = seq + 1
            writer.writerow([seq, t_us, ax, ay, az])
            rows += 1
            last_time_us = t_us
            if rows % 1000 == 0:
                print('{} sampel diterima'.format(rows))

    print('Selesai:', output_file)
    print('Data tersimpan:', rows, '/', TOTAL_SAMPLES)
    if last_time_us is not None:
        print('Timestamp akhir: {:.3f} detik'.format(last_time_us / 1_000_000))
    if errors or rows != TOTAL_SAMPLES or not ended:
        print('PERINGATAN: Ada data hilang/tidak terbaca (indikasi:', errors, ')')
    else:
        print('Semua sampel diterima.')


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print('Menghubungkan ke ESP32 melalui', PORT)
    with serial.Serial(PORT, BAUD, timeout=0.5, write_timeout=2) as ser:
        handshake(ser)
        print('ESP32 siap. Target {} Hz, {} detik / rekaman.'.format(
            FS, DURATION_SECONDS
        ))
        while True:
            name = input('\nNama file (tanpa .csv; ENTER kosong untuk keluar): ').strip()
            if not name:
                break
            # Jangan membiarkan nama pengguna keluar dari folder data.
            name = Path(name).name
            if not name or name in ('.', '..'):
                print('Nama file tidak valid.')
                continue
            filename = name if name.lower().endswith('.csv') else name + '.csv'
            output_file = DATA_DIR / filename
            if output_file.exists():
                print('File sudah ada; masukkan nama lain agar tidak menimpa data.')
                continue
            print('Mulai akuisisi ->', output_file)
            try:
                capture(ser, output_file)
            except Exception as ex:
                print('Perekaman terganggu:', ex)
                print('Periksa file CSV (mungkin parsial), lalu reset ESP32 bila perlu.')
                break
    print('Logger ditutup.')


if __name__ == '__main__':
    main()
