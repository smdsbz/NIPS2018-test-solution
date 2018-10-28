from multiprocessing import Process, Pipe
from time import sleep


def echo(pipe):
    while True:
        val = pipe.recv()
        sleep(0.2)
        print('echoing:', val)
        if val == 'stop':
            pipe.close()
            break


if __name__ == '__main__':
    writer, reader = Pipe()
    p = Process(target=echo, args=(reader,))
    p.start()
    for st in range(10):
        writer.send(st)
    writer.send('stop')
    print('stop signal sent!')
    p.join()

