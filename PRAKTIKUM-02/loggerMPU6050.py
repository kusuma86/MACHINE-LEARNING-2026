# main.py -- simpan pada ESP32 dengan firmware MicroPython
# Menunggu perintah dari laptop: PING atau START,<jumlah_sampel>
from machine import Pin, I2C
import sys
import time

FS = 200                  # Target sampel/detik
PERIOD_US = 1_000_000 // FS

# Untuk ESP32 DevKit klasik: SDA=21, SCL=22
sensor = I2C(0, sda=Pin(21), scl=Pin(22), freq=400_000)
addresses = sensor.scan()
if 0x68 in addresses:
    MPU = 0x68
elif 0x69 in addresses:
    MPU = 0x69
else:
    raise OSError('MPU6050 tidak ditemukan; periksa kabel SDA/SCL/VCC/GND')

# Reset sleep -> clock PLL, DLPF aktif, output rate 200 Hz,
# accel range +/- 4g (8192 LSB/g)
sensor.writeto_mem(MPU, 0x6B, b'\x01')
time.sleep_ms(100)
sensor.writeto_mem(MPU, 0x1A, b'\x02')  # DLPF
sensor.writeto_mem(MPU, 0x19, b'\x04')  # 1 kHz / (1+4) = 200 Hz
sensor.writeto_mem(MPU, 0x1C, b'\x08')  # +/- 4g


def to_signed16(hi, lo):
    x = (hi << 8) | lo
    return x - 65536 if x & 0x8000 else x


def acquire(n):
    # BEGIN memberi tahu laptop bahwa akuisisi benar-benar dimulai.
    print('BEGIN,{}'.format(n))
    start = time.ticks_us()
    due = start
    for index in range(n):
        wait_us = time.ticks_diff(due, time.ticks_us())
        if wait_us > 0:
            time.sleep_us(wait_us)

        stamp = time.ticks_diff(time.ticks_us(), start)
        raw = sensor.readfrom_mem(MPU, 0x3B, 6)
        x = to_signed16(raw[0], raw[1])
        y = to_signed16(raw[2], raw[3])
        z = to_signed16(raw[4], raw[5])
        print('D,{},{},{},{},{}'.format(index, stamp, x, y, z))
        due = time.ticks_add(due, PERIOD_US)

    print('END,{}'.format(n))


print('READY,{},{}'.format(FS, hex(MPU)))
while True:
    try:
        command = sys.stdin.readline().strip()
        if command == 'PING':
            print('PONG')
        elif command.startswith('START,'):
            try:
                n = int(command.split(',', 1)[1])
                if n < 1 or n > 100000:
                    raise ValueError
                acquire(n)
            except ValueError:
                print('ERROR,INVALID_COUNT')
            except OSError:
                print('ERROR,I2C_READ')
        elif command:
            print('ERROR,UNKNOWN_COMMAND')
    except KeyboardInterrupt:
        # Saat pengembangan di Thonny, Ctrl+C tetap bisa menghentikan program.
        raise

