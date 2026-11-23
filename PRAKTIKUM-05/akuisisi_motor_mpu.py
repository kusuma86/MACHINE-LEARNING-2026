"""
ESP32 + L298N + motor DC ber-encoder + MPU6050
Akuisisi data getaran untuk regresi RPM (MicroPython).

Disesuaikan dari kode kontrol motor yang sudah berjalan:
  ENA 25, IN1 27, IN2 26
  Encoder A 18, B 19 (PPR=330)
  MPU6050 SDA 21, SCL 22 (3V3, GND)
  PWM 1000 Hz; duty() 0..1023; Kp=1.0, Ki=20.3, Kd=0; MIN_PWM=200

Per setpoint: stabilisasi (kontrol aktif), akuisisi getaran 200 Hz
ke RAM, motor berhenti, CSV disimpan ke flash ESP32.
Label ML = rpm_actual (encoder), fitur ML = getaran MPU6050.

PERHATIAN:Hindari benda yang bisa tersangkut.
"""

from machine import Pin, PWM, I2C, disable_irq, enable_irq
import time
import struct
import gc

# ====================== KONFIGURASI ======================
# Pin sesuai program motor yang sudah teruji
PIN_ENA = 25
PIN_IN1 = 27
PIN_IN2 = 26
PIN_ENC_A = 18
PIN_ENC_B = 19
PIN_SDA = 21
PIN_SCL = 22

# Sesuai program motor yang sudah teruji
PPR = 330                    # Hitungan rising edge encoder A per putaran output
KP = 1.0
KI = 20.3
KD = 0.0
MAX_INTEGRAL = 1000.0
MIN_PWM = 200
PWM_FREQ = 1000              # Hz; sama dengan kode motor yang berfungsi
CONTROL_PERIOD_MS = 50      # 50 ms

# Percobaan gabungan (boleh diganti sesuai kebutuhan)
RPM_TARGETS = (70, 60, 100, 90, 50, 100, 110, 120, 80, 75)
REPEATS = 1
STABILIZE_S = 5              # Motor mencapai RPM sebelum mulai logger
RECORD_S = 10                # Durasi getaran per target
SAMPLE_HZ = 200              # Pembacaan MPU6050 (target)
ACCEL_RANGE_G = 4            # +/-4 g (pilih 4 atau 8)

if ACCEL_RANGE_G not in (4, 8):
    raise ValueError('ACCEL_RANGE_G harus 4 atau 8')
if 1_000_000 % SAMPLE_HZ != 0:
    raise ValueError('SAMPLE_HZ harus membagi 1.000.000 dengan tepat')

SAMPLE_PERIOD_US = 1_000_000 // SAMPLE_HZ
ACC_SCALE = 8192.0 if ACCEL_RANGE_G == 4 else 4096.0

# Buffer RAM padat: t_us:uint32, ax/ay/az:int16, rpm:float32, pwm:uint16
# 16 byte x 200 Hz x 10 s = 32.000 byte per sesi
PACK_FMT = '<IhhhfH'
PACK_SIZE = struct.calcsize(PACK_FMT)

# ===================== INITIALISASI =====================
enc_a = Pin(PIN_ENC_A, Pin.IN)   # Tanpa PULL_UP: mengikuti program teruji
enc_b = Pin(PIN_ENC_B, Pin.IN)
forward_pin = Pin(PIN_IN1, Pin.OUT)
backward_pin = Pin(PIN_IN2, Pin.OUT)
enable_pwm = PWM(Pin(PIN_ENA), freq=PWM_FREQ)

# Motor OFF sebelum mengakses I2C
forward_pin.value(0)
backward_pin.value(0)
enable_pwm.duty(0)

i2c = I2C(0, sda=Pin(PIN_SDA), scl=Pin(PIN_SCL), freq=400_000)
found = i2c.scan()
if 0x68 in found:
    MPU_ADDR = 0x68
elif 0x69 in found:
    MPU_ADDR = 0x69
else:
    raise OSError('MPU6050 tidak terdeteksi. Alamat I2C: %s' % found)

# MPU6050: keluar dari sleep; filter digital; data-rate 200 Hz;
# accelerometer +/-8 g atau +/-4 g.
i2c.writeto_mem(MPU_ADDR, 0x6B, b'\x00')  # PWR_MGMT_1
time.sleep_ms(100)
i2c.writeto_mem(MPU_ADDR, 0x1A, b'\x02')  # DLPF_CFG=2 (~94 Hz acc)
i2c.writeto_mem(MPU_ADDR, 0x19, b'\x04')  # sample-rate divider=4 -> 200 Hz
i2c.writeto_mem(MPU_ADDR, 0x1C,
                b'\x10' if ACCEL_RANGE_G == 8 else b'\x08')

# ====================== ENCODER / PID ======================
encoder_value = 0
last_encoder = 0
last_control_ms = time.ticks_ms()
integral = 0.0
last_error = 0.0
rpm_actual = 0.0
pwm_val = 0


def update_encoder(pin):
    global encoder_value
    # SAMA seperti kode motor yang sudah berjalan.
    if enc_a.value() > enc_b.value():
        encoder_value += 1
    else:
        encoder_value -= 1


enc_a.irq(trigger=Pin.IRQ_RISING, handler=update_encoder)


def stop_motor():
    global pwm_val
    pwm_val = 0
    enable_pwm.duty(0)
    forward_pin.value(0)
    backward_pin.value(0)


def reset_pid():
    global last_encoder, last_control_ms, integral, last_error, rpm_actual
    irq_state = disable_irq()
    last_encoder = encoder_value
    enable_irq(irq_state)
    last_control_ms = time.ticks_ms()
    integral = 0.0
    last_error = 0.0
    rpm_actual = 0.0


def pid_step(rpm_target):
    global last_encoder, last_control_ms
    global integral, last_error, rpm_actual, pwm_val

    now = time.ticks_ms()
    raw_dt = time.ticks_diff(now, last_control_ms)
    if raw_dt < CONTROL_PERIOD_MS:
        return

    dt = raw_dt / 1000.0
    last_control_ms = now

    irq_state = disable_irq()
    count = encoder_value
    enable_irq(irq_state)
    delta_encoder = count - last_encoder
    last_encoder = count
    rpm_actual = (delta_encoder / PPR) * (60.0 / dt)

    # Logika PID, PWM, dan arah DIAMBIL dari kode yang sudah berfungsi.
    error = rpm_target - rpm_actual
    integral += error * dt
    integral = min(max(integral, -MAX_INTEGRAL), MAX_INTEGRAL)
    derivative = (error - last_error) / dt
    output = KP * error + KI * integral + KD * derivative
    last_error = error

    pwm_val = min(max(int(abs(output)), 0), 1023)
    if abs(error) > 1 and pwm_val > 0:
        pwm_val = max(pwm_val, MIN_PWM)

    enable_pwm.duty(pwm_val)

    # Arah motor persis seperti kode yang berhasil.
    if output >= 0:
        forward_pin.value(0)
        backward_pin.value(1)
    else:
        forward_pin.value(1)
        backward_pin.value(0)


def stabilize(rpm_target):
    t_start = time.ticks_ms()
    last_print = t_start
    while time.ticks_diff(time.ticks_ms(), t_start) < STABILIZE_S * 1000:
        pid_step(rpm_target)
        now = time.ticks_ms()
        if time.ticks_diff(now, last_print) >= 1000:
            print('  target=%d | encoder=%.1f RPM | PWM=%d' %
                  (rpm_target, rpm_actual, pwm_val))
            last_print = now
        time.sleep_ms(1)


def acquire(rpm_target):
    # Tidak ada print atau penulisan CSV selama sampling.
    gc.collect()
    n_samples = RECORD_S * SAMPLE_HZ
    data = bytearray(n_samples * PACK_SIZE)
    start_us = time.ticks_us()
    next_us = start_us
    late_count = 0
    max_lateness_us = 0

    for n in range(n_samples):
        # PID tetap berjalan setiap ~50 ms, walau sensor dibaca ~200 Hz.
        while time.ticks_diff(next_us, time.ticks_us()) > 0:
            pid_step(rpm_target)
            time.sleep_us(100)

        pid_step(rpm_target)
        t_now = time.ticks_us()
        t_rel = time.ticks_diff(t_now, start_us)

        # Register 0x3B..0x40: 6 byte accelerometer X,Y,Z.
        ax, ay, az = struct.unpack('>hhh',
                                   i2c.readfrom_mem(MPU_ADDR, 0x3B, 6))
        struct.pack_into(PACK_FMT, data, n * PACK_SIZE,
                         t_rel, ax, ay, az, rpm_actual, pwm_val)

        next_us = time.ticks_add(next_us, SAMPLE_PERIOD_US)
        lateness = time.ticks_diff(time.ticks_us(), next_us)
        if lateness > 0:
            late_count += 1
            if lateness > max_lateness_us:
                max_lateness_us = lateness
            if lateness >= SAMPLE_PERIOD_US:
                # Hindari mengejar timestamp yang sudah tertinggal jauh.
                next_us = time.ticks_add(time.ticks_us(), SAMPLE_PERIOD_US)

    return data, n_samples, late_count, max_lateness_us


def save_csv(filename, run_id, stage_id, rpm_target, data, n_samples):
    prev_t = None
    with open(filename, 'w') as f:
        f.write('t_us,dt_us,sample_index,run,stage,setpoint_rpm,'
                'rpm_actual,pwm,ax_g,ay_g,az_g\n')
        for i in range(n_samples):
            t_us, ax, ay, az, rpm, duty = struct.unpack_from(
                PACK_FMT, data, i * PACK_SIZE)
            dt_us = 0 if prev_t is None else t_us - prev_t
            prev_t = t_us
            f.write('%d,%d,%d,%d,%d,%d,%.4f,%d,%.6f,%.6f,%.6f\n' % (
                t_us, dt_us, i, run_id, stage_id, rpm_target,
                rpm, duty, ax / ACC_SCALE, ay / ACC_SCALE, az / ACC_SCALE))


def main():
    print('MPU6050 ditemukan: 0x%02X' % MPU_ADDR)
    print('PWM=%d Hz, duty=0..1023, PPR=%d' % (PWM_FREQ, PPR))
    print('MPU=%d Hz, +/- %dg, PID=%d ms' %
          (SAMPLE_HZ, ACCEL_RANGE_G, CONTROL_PERIOD_MS))
    print('Setpoint:', RPM_TARGETS)
    print('1 file CSV untuk tiap tahap. Hindari menyimpan file berbeda'
          ' dengan nama yang sama pada percobaan ulang.')

    try:
        for run_id in range(1, REPEATS + 1):
            for stage_id, rpm_target in enumerate(RPM_TARGETS, 1):
                print('\nRun %d, tahap %d: target %d RPM' %
                      (run_id, stage_id, rpm_target))
                stop_motor()
                time.sleep_ms(300)
                reset_pid()

                # Sama dengan arah awal di kode asli.
                forward_pin.value(0)
                backward_pin.value(1)
                stabilize(rpm_target)
                print('  Merekam %d detik...' % RECORD_S)
                data, count, late, max_late = acquire(rpm_target)

                # Simpan CSV setelah motor berhenti; tidak mengganggu sampling.
                stop_motor()
                filename = 'motor_r%02d_s%02d_sp%03d.csv' % (
                    run_id, stage_id, rpm_target)
                save_csv(filename, run_id, stage_id, rpm_target, data, count)
                print('  Tersimpan: %s (%d sampel)' % (filename, count))
                print('  RPM akhir=%.1f, keterlambatan=%d, terburuk=%d us' %
                      (rpm_actual, late, max_late))
                data = None
                gc.collect()

        print('\nAkuisisi selesai. File dapat diunduh melalui Thonny.')
    finally:
        stop_motor()
        enc_a.irq(handler=None)
        print('Motor OFF')


if __name__ == '__main__':
    main()

