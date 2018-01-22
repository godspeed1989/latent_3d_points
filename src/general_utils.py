import numpy as np
from ply_file_internal import PlyData, PlyElement

def rand_rotation_matrix(deflection=1.0, seed=None):
    '''Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi     # For direction of pole deflection.
    z = z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

if __name__ == '__main__':
    def read_ply(filename):
        """ read XYZ point cloud from filename PLY file """
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pc_array = np.array([[x, y, z] for x,y,z in pc])
        return pc_array

    def write_ply(points, filename, text=True):
        """ input: Nx3, write points to filename as PLY format. """
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(filename)
    # data in
    filename = 'printer_3d'
    pc_in = read_ply(filename + '.ply')
    pc_in = np.reshape(pc_in, (1,-1,3))
    # rotate
    r_rotation = rand_rotation_matrix()
    print(r_rotation)
    # data out
    pc_out = pc_in.dot(r_rotation)
    write_ply(pc_out[0], 'rot_' + filename + '.ply')
    # check radius
    v1 = np.sqrt(pc_out[0][:,0]*pc_out[0][:,0] + pc_out[0][:,1]*pc_out[0][:,1] + pc_out[0][:,2]*pc_out[0][:,2])
    v2 = np.sqrt(pc_in[0][:,0]*pc_in[0][:,0] + pc_in[0][:,1]*pc_in[0][:,1] + pc_in[0][:,2]*pc_in[0][:,2])
    print(v1 - v2)
