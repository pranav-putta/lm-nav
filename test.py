import multiprocessing as mp


def func():
    try:
        a = 1
    except KeyboardInterrupt:
        print("Interrupted")

parent_con, _ = mp.Pipe()
ps = mp.Process(target=func)
ps.start()
while ps.is_alive():
    ps.join(0)
b = 3
print(b)
