import numpy as np
import open3d as o3d

def print_points(node, layers=[], newlayer=0):
    if node == None:
        return
    layers.append(newlayer)
    for i in range(0, len(node.children)):
        print_points(node.children[i], layers.copy(), i)
    print('{} | {}'.format(layers, node.points[:5]))

def create_sphere(box, sphere_density):
    (cx,cy,cz, r, resolution) = (box.center_point[0], box.center_point[1], box.center_point[2], box.cube_dim_half, int(box.cube_dim_half / sphere_density))

    theta = np.radians(np.linspace(0, 360, resolution+1))
    phi = np.radians(np.linspace(0, 360, resolution+1))

    x = cx + r * np.einsum("i,j->ij", np.cos(phi), np.sin(theta))
    y = cy + r * np.einsum("i,j->ij", np.sin(phi), np.sin(theta))
    z = cz + r * np.einsum("i,j->ij", np.ones(len(theta)), np.cos(theta))
    xyz = np.array([x.flatten(), y.flatten(), z.flatten()])
    return xyz.T

def create_bounds_sphere(box, sphere_density):
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(create_sphere(box, sphere_density))
    color = np.random.rand(1,3)[0]
    geom.paint_uniform_color(color)
    return (geom, color)

def create_bounds_cube(box, color, cube_density):
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(box.bounds_geometry(cube_density))
    geom.paint_uniform_color(color)
    return geom

def create_geometry(node, cube_density, sphere_density, geometries=[], root=True):
    if node == None:
        return
    color = [0, 0, 0]
    if not root:
        (sphere_geom, color) = create_bounds_sphere(node.box, sphere_density)
        geometries.append(sphere_geom)
    geometries.append(create_bounds_cube(node.box, color, cube_density))
    
    for child in node.children:
        child_geom = create_geometry(child, cube_density, sphere_density, geometries, False)

    if len(node.points) < 1:
        return
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(node.points)
    if not np.equal(color,[0, 0, 0]).all(0):
        geom.paint_uniform_color(color)
    geometries.append(geom)