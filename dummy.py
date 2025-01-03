with open("performance.txt", "r") as f:
    n = int(f.readline()[6:])
    act = f.readline()[14:].strip()
    opt = f.readline()[13:].strip()
    acc = float(f.readline()[12:-2]) / 100
    l = float(f.readline()[8:])
    time = float(f.readline()[8:-3])

    print(n)
    print(act)
    print(opt)
    print(acc)
    print(l)
    print(time)