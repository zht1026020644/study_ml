import math
import time

if __name__ == "__main__":
    a = 2
    b = 100000
    # 方法1
    t = time.time()
    p = [p for p in range(a, b) if 0 not in [p % d for d in range(2, int(math.sqrt(p)) + 1)]]
    print(time.time() - t)
    print(p)

    #
