import time

for i in range(1000):
    print(int(time.time() * 1000))
    time.sleep(0.001)
    print(int(time.time() * 1000))
