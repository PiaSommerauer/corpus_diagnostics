import multiprocessing

def worker(procnum, qu):
    ret = qu.get()
    print(str(procnum) + ' represent!')
    ret.append(procnum)
    qu.put(ret)


if __name__ == '__main__':
    jobs = []
    qu = multiprocessing.Queue()
    ret = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,qu))
        jobs.append(p)
        p.start()

    results = []
    for proc in jobs:
        proc.join()
        ret = qu.get()
        results.append(ret)
    print(results)