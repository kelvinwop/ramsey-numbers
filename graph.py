import numpy as np
import time
from tqdm import tqdm
import sys

class InputGraphTooSmall(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Graph:
    lookup_table = np.zeros(33554432, dtype=int)  # set to 2x largest mult value in hash function
    lt_stats = {"additions": 0, "lookups": 0}

    def __init__(self, size, random=True, error_check=False):
        if random:
            self.data = np.tril(np.random.randint(0, high=2, size=(size,size)), k=-1)
            self.data = np.add(np.add(self.data, np.transpose(self.data)), np.identity(size, dtype=np.int32))
            self.data_inverse = np.add(np.subtract(np.ones((size, size), dtype=np.int32), self.data), np.identity(size, dtype=np.int32))
        else:
            self.data = np.zeros((size, size), dtype=np.int32)
            self.data_inverse = np.ones((size, size), dtype=np.int32)
        self.error_trace = ""
        self.error_check = error_check
    
    def __str__(self):
        return str(self.data)

    def remove_elem(self, elem, data=None, copy=True):
        """removes a row and column by index 'elem'
        you can remove multiple by specifying an array instead of a number
        """
        if isinstance(data, type(None)):
            data = self.data
        if copy:
            data = data.copy()
        if data.size == 1:
            raise InputGraphTooSmall(str(data))
        data = np.delete(data, elem, 0)
        data = np.delete(data, elem, 1)
        return data
    
    def print_all(self):
        d = str(self.data)
        di = str(self.data_inverse)
        print("Data: \n" + d + "\n\nData inverse:\n" + di)

    def swap(self, data, idx_a, idx_b):
        """swaps two rows and columns by the given indices"""
        data[[idx_a, idx_b]] = data[[idx_b, idx_a]]
        data[:,[idx_a, idx_b]] = data[:,[idx_b, idx_a]]

    def align_ones(self, data, copy=True, debug=False):
        """aligns the top connections [1]s to the left and returns the smaller matrix
        if copy is set to true [default], the given data will be deep copied

        TODO: maybe instead of the top and left sides we do the bot and right for O(n) -> O(c)
        """
        if copy:
            data = data.copy()
        last_idx = np.shape(data)[0] - 1
        idx_a = last_idx
        idx_b = 0
        while idx_a > idx_b:
            if data[last_idx, idx_a] == 1:  # if bottom right is 1, keep moving
                idx_a -= 1
                continue
            if data[last_idx, idx_b] == 0:  # if bottom left is 0, keep moving
                idx_b += 1
                continue
            self.swap(data, idx_a, idx_b)
            idx_a -= 1
            idx_b += 1
        amt_ones = np.sum(data, axis=0)[last_idx]
        output = data[(last_idx - amt_ones + 1):, (last_idx - amt_ones + 1):]
        if debug:
            print("align_ones output:\n" + str(output) + "\nalign_ones input:\n" + str(data))
        return output, data

    def eliminate_row(self, data=None, copy=True, fast=True, debug=False):
        """attempts to remove a vertex from the problem to reduce the difficulty.
        if copy is set to true [default], the given data is deep copied
        if fast is set to true [default], the algorithm will try to eliminate the sparsest row

        returns the largest clique the row was a part of and the remaining data
        """
        assert data.size > 1
        if isinstance(data, type(None)):
            data = self.data
        if copy:
            data = data.copy()
        if fast:
            vertex_with_fewest_connections = np.argmin(np.sum(data, axis=0))
            self.swap(data, np.shape(data)[0]-1, vertex_with_fewest_connections)
            if self.error_check:
                self.error_trace += "\n\n"
                self.error_trace += "eliminate_row init with fast, swapping %s and %s" % (np.shape(data)[0]-1, vertex_with_fewest_connections)
            
        try:
            # pre_subset is the graph where all 1s are moved to the bottom right
            pre_subset, debugging_matrix = self.align_ones(data)
            chopped = False
            if pre_subset.size > 1:
                # the row of all 1s is removed from the cutout group
                # note this is not related to the data being passed back (directly)
                subset = self.remove_elem(np.shape(pre_subset)[0]-1, data=pre_subset)
                chopped = True
            else:
                subset = pre_subset
        except Exception as exc:
            raise type(exc)(str(exc) + "\neliminate_row subset:\n" + str(pre_subset) + "\nAND data\n" + str(data))
        if debug:
            print("elim row:\n", data)
            time.sleep(0.5)

        try:
            # 1 is added since we removed the row that was connected to everything
            largest_clique = self.__get_largest_clique(subset)
            if chopped:
                largest_clique += 1
        except Exception as exc:
            raise type(exc)(str(exc) + "\neliminate_row subset\n" + str(subset))

        try:
            # we remove the row that we have evaluated
            new_set = self.remove_elem(np.shape(data)[0]-1, data=data)
        except Exception as exc:
            raise type(exc)(str(exc) + "\neliminate_row data\n" + str(data))

        if self.error_check:
            self.error_trace += "\n\n"
            self.error_trace += "in eliminate_row,\n"
            self.error_trace += "largest_clique found in pre_subset: %s\n" % largest_clique
            self.error_trace += "in data: \n%s\n" % data
            self.error_trace += "data after aligning ones: \n%s\n" % debugging_matrix
            self.error_trace += "pre_subset: \n%s\n" % pre_subset 
            self.error_trace += "subset: \n%s\n" % subset 
            self.error_trace += "new set: \n%s\n" % new_set 
        return largest_clique, new_set

    def __get_largest_clique(self, data=None, copy=True, debug=False):
        """attempts to find the largest number of fully connected nodes
        if copy is set to true [default], the given data is deep copied

        returns the largest clique size
        """
        if isinstance(data, type(None)):
            data = self.data
        if copy:
            data = data.copy()
        if data.size == 0:
            return 0
        shape = np.shape(data)
        assert shape[0] == shape[1], "got: " + str(data)  # squares only >:(

        connections = np.sum(data)
        if connections == 0 or data.size == 1:  # one node
            return 1
        if connections - shape[0] < 2:
            return 1
        if connections - shape[0] < 6:  # smaller than smallest 3x3 clique
            return 2
        if data.size == connections:  # the graph is fully connected
            return shape[0]

        hash_val = None
        if data.size <= 25:
            # lookup table
            hash_val = self.hash(data)
            if Graph.lookup_table[hash_val] != 0:
                Graph.lt_stats["lookups"] += 1
                return Graph.lookup_table[hash_val]
            

        if debug:
            print("getlargestclique:\n", data)

        try:
            largest_in_row, new_graph = self.eliminate_row(data=data)
        except Exception as exc:
            raise type(exc)(str(exc) + "\nget_largest_clique data\n" + str(data))

        if debug:
            print("getlargestclique")
            print(data)
            print("largest in row: %s" % largest_in_row)
            time.sleep(0.5)

        try:
            result = max(largest_in_row, self.__get_largest_clique(new_graph, copy=False))

            # hash the result to save time in the future
            if hash_val is not None:
                Graph.lt_stats["additions"] += 1
                Graph.lookup_table[hash_val] = result
            return result
        except Exception as exc:
            raise type(exc)(str(exc) + "\nget_largest_clique new_graph [return]\n" + str(new_graph))
        
    def get_largest_clique(self):
        norm_data = self.__get_largest_clique(self.data)
        inv_data = self.__get_largest_clique(self.data_inverse)
        if norm_data >= inv_data:
            if norm_data == 2 and self.data.size == 36:
                # catastrophic error
                with open("error.txt", "w+") as f:
                    f.write(self.error_trace)
                sys.exit(300)
            return norm_data, self.data
        if inv_data > norm_data:
            if inv_data == 2 and self.data_inverse.size >= 36:
                # catastrophic error
                with open("error.txt", "w+") as f:
                    f.write(self.error_trace)
                sys.exit(301)
            return inv_data, self.data_inverse

    def hash(self, data):
        if data.size == 1:
            return data[0]
        if data.size == 4:
            return np.sum(np.multiply(data, np.array([[1,2],[4,8]], dtype=np.int32)))
        if data.size == 9:
            return np.sum(np.multiply(data, np.array([[1,2,4],[8,16,32],[64,128,256]], dtype=np.int32)))
        if data.size == 16:
            return np.sum(np.multiply(data, np.array([[1,2,4,8],[16,32,64,128],[256,512,1024,2048],[4096,8192,16384,32768]], dtype=np.int32)))
        if data.size == 25:
            return np.sum(np.multiply(
                data,
                np.array(
                    [[1,2,4,8,16],
                    [32,64,128,256,512],
                    [1024,2048,4096,8192,16384],
                    [32768,65536,131072,262144,524288],
                    [1048576,2097152,4194304,8388608,16777216]], dtype=np.int32
                )
            ))
        raise Exception("data size not supported")