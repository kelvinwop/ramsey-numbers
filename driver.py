from tqdm import tqdm
import sys
import numpy as np
import graph
import json
import time

profile=False

def main():
    np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
    print("working")
    for g in range(1,19):
        ss = 999999
        se = None
        found = dict()
        found["total"] = 0
        times = []
        last_i_written = 0
        for i in tqdm(range(1000000)):  # tqdm for progress bars
            a = graph.Graph(g)

            start_time = time.time()
            lc, example = a.get_largest_clique()
            end_time = time.time()

            times.append(end_time-start_time)
            if str(lc) in found:
                found[str(lc)] += 1
            else:
                found[str(lc)] = 1
            found["total"] += 1

            if ss > lc or i - last_i_written > 1000:
                if ss > lc:
                    ss = lc
                    se = example
                last_i_written = i
                my_str = "smallest largest clique found: %s" % ss
                my_str += "\nshown here:\n" + str(se)
                my_str += "\n" + json.dumps(found, sort_keys=True, indent=4, separators=(',', ': '))
                my_str += "\nhashtable(array) stats\n" + json.dumps(a.lt_stats, sort_keys=True, indent=4, separators=(',', ': '))
                with open("output.txt", "w+") as f:
                    f.write(my_str)
        my_str = "smallest largest clique found: %s" % ss
        my_str += "\nshown here:\n" + str(se)
        my_str += "\n" + json.dumps(found, sort_keys=True, indent=4, separators=(',', ': '))
        my_str += "\nhashtable(array) stats\n" + json.dumps(a.lt_stats, sort_keys=True, indent=4, separators=(',', ': '))
        #print(my_str)
        with open("output%s.txt" % g, "w+") as f:
            f.write(my_str)

        print("nodes: %s, average time: %s" % (g, str(np.average(times))))

if profile:
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    import pycallgraph
    with PyCallGraph(output=GraphvizOutput()):
        main()
else:
    main()