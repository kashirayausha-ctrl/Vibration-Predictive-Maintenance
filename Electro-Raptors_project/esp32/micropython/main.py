from machine import ADC, Pin
import time

# Connect the piezo conditioning circuit to an ADC-capable input.
PIEZO_ADC_PIN = 34
SAMPLE_INTERVAL_US = 10_000  # 100 Hz

piezo = ADC(Pin(PIEZO_ADC_PIN))
piezo.atten(ADC.ATTN_11DB)
piezo.width(ADC.WIDTH_12BIT)

print("piezo")
next_sample = time.ticks_us()

while True:
    now = time.ticks_us()
    if time.ticks_diff(now, next_sample) >= 0:
        next_sample = time.ticks_add(next_sample, SAMPLE_INTERVAL_US)
        print(piezo.read())
    else:
        time.sleep_us(500)
