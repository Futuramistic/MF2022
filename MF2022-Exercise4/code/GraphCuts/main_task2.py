from graph_cut_controller import GraphCutController
from graph_cut import *


def task_2_1():
    # Source weights: column[0]
    # Target weights: column[1]

    n_nodes = 3

    graph = GraphCut(n_nodes,2)
    unary = np.array([ [4,9],
                         [7,7],
                         [8,5]])

    # Each row represents two edges:
    # e[0] -> e[1] with cap e[3]
    # e[0] <- e[1] with cap e[4]
    pairwise = np.array([
        [0,1,0,3,2,0],
        [1,2,0,5,1,0]
    ])

    graph.set_unary(unary)
    graph.set_pairwise(pairwise)

    maxFlow = graph.minimize()
    labels = np.array(graph.get_labeling())
    nodes = np.arange(n_nodes)

    source_partition = nodes[~labels]
    target_partition = nodes[labels]

    print(f'Max flow value: {maxFlow}')
    print(f'Nodes in source partition: {source_partition}')
    print(f'Nodes in target partition: {target_partition}')

def task_2_2():
    GraphCutController()

def main():
    task_2_1()
    task_2_2()


if __name__ == '__main__':
    main()
