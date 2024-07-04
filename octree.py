import numpy as np
import time
from boundary_box import BoundaryBox


class Octree:
    def __init__(self, box, points=[], parent=None, layers_left=2):
        self.box = box
        self.points = points
        self.parent = parent
        self.layers_left = layers_left
        self.children = []
        for _ in range(8):
            self.children.append(None)

    def Build(self, prev_layers=[], layer_index=0, debug=True, remove_points_outside_sphere=True):
        if self.layers_left < 1:
            return
                
        new_length = self.box.cube_dim_half / 2

        # if node isn't root, created cubes are shrunk to be inside the sphere
        if len(prev_layers) > 0:
            new_length = new_length / 1.7321

        prev_layers.append(layer_index)
        if debug:
            print("-")
            print(prev_layers)

        octants = []
        octants.append(BoundaryBox([self.box.center_point[0] - new_length, self.box.center_point[1] - new_length, self.box.center_point[2] - new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] + new_length, self.box.center_point[1] - new_length, self.box.center_point[2] - new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] - new_length, self.box.center_point[1] + new_length, self.box.center_point[2] - new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] + new_length, self.box.center_point[1] + new_length, self.box.center_point[2] - new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] - new_length, self.box.center_point[1] - new_length, self.box.center_point[2] + new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] + new_length, self.box.center_point[1] - new_length, self.box.center_point[2] + new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] - new_length, self.box.center_point[1] + new_length, self.box.center_point[2] + new_length], new_length))
        octants.append(BoundaryBox([self.box.center_point[0] + new_length, self.box.center_point[1] + new_length, self.box.center_point[2] + new_length], new_length))

        points_to_remove_indexes = []
        points_in_children_nodes = []
        points_in_children_is_empty = True
        for _ in range(8):
            points_in_children_nodes.append([])

        m = 0
        printLim = 10000
        start = time.time()
        for i in range(0, len(self.points)):
            point = self.points[i]
            m = m + 1
            if m % printLim == 0:
                end = time.time()

                if debug:
                    print('{} | {:.3f} s'.format(m, end - start))
                start = end
                if printLim <= 800000:
                    printLim = printLim * 10
            for ii in range(0, len(octants)):
                contains = octants[ii].contains(point)
                if contains == 2:
                    points_to_remove_indexes.append(i)
                    points_in_children_nodes[ii].append(point)
                    if points_in_children_is_empty:
                        points_in_children_is_empty = False
                    break
                elif contains == 1:
                    if remove_points_outside_sphere:
                        points_to_remove_indexes.append(i)
                    break
        if debug:
            end = time.time()
            print('{} | {:.3f} s'.format(m, end - start))
        
        self.points = np.delete(self.points, points_to_remove_indexes, axis=0)
        for i in range(0, len(points_in_children_nodes)):
            if not points_in_children_is_empty:
                self.children[i] = Octree(octants[i], points_in_children_nodes[i], self, self.layers_left-1)
            if len(points_in_children_nodes[i]) > 0:
                self.children[i].Build(prev_layers.copy(), i, debug, remove_points_outside_sphere)

    def find(self, point):
        for child in self.children:
            if child == None:
                continue
            if child.find(point):
                return True
        
        if self.box.contains(point) != 2:
            return False
        
        return any(np.equal(self.points,point).all(1))
    
    def add(self, point):
        for child in self.children:
            if child == None:
                continue
            if child.add(point):
                return True
        
        if self.box.contains(point) != 2:
            return False
        
        self.points = np.vstack((self.points, point))
        return True