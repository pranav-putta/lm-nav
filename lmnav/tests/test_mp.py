import multiprocessing as mp
import time

def foo(q):
    for i in range(3):
        time.sleep(1)
        print('.', end='')
        q.put(1)
    q.put('hello')

def main():
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    while q:
        print(q.get())

if __name__ == '__main__':
    main()

