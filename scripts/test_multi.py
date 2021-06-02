import multiprocessing


def test_process(qu, v):
    d = dict()
    search = [1,3,5]
    for s in search:
        if v == s:
            d[s]=v
    qu.put(d)


if __name__ == '__main__':
    jobs = []

    qu = multiprocessing.Queue()

    output = []
    my_list = []
    print('start multi')
    for i in range(5):
        p = multiprocessing.Process(target=test_process, args=(qu, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        out = qu.get()
        output.append(out)
        proc.join()


    print(output)