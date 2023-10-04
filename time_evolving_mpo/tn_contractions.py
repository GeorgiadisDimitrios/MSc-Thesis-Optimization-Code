import numpy as np
import tensornetwork as tn

def return_arrays_1():
    list_of_matrices_1 = []
    matrix_1 = np.array([[0,1],[2,3]])
    matrix_2 = np.array([[4,5],[6,7]])
    node_1 = tn.Node(matrix_1)
    node_2 = tn.Node(matrix_2)
    list_of_matrices_1.append(node_1)
    list_of_matrices_1.append(node_2)
    return list_of_matrices_1

def return_arrays_2():
    list_of_matrices_2 = []
    matrix_1 = np.array([[10,11],[12,13]])
    matrix_2 = np.array([[14,15],[16,17]])
    node_1 = tn.Node(matrix_1)
    node_2 = tn.Node(matrix_2)
    list_of_matrices_2.append(node_1)
    list_of_matrices_2.append(node_2)
    return list_of_matrices_2

nodes_1 = return_arrays_1()
nodes_2 = return_arrays_2()

test_node_1 = nodes_1[0]
test_node_2 = nodes_2[0]

test_node_1[0] ^ test_node_2[1]


result = test_node_1 @ test_node_2

print(result.tensor)


