import numpy as np
from sympy.tensor.array import derive_by_array #Symbolic gradient calculation
from sympy.abc import x,y,z #Defining symbolic parameters for sympy and for the function so that I can calc the gradient at any point
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
# from mpl_toolkits import mplot3d
from sphere import Sphere

class Flow:
    def __init__(self, func, scatter = None): #flow class designed to "flow" a certain scatter according to some function func
        self.func = func
        self.grad = derive_by_array(func, (x,y,z))
        if scatter == None:
            #optional - add your own scatter (not just sphere). notice that by default the nubmer of points taken on the sphere is 10k
            self.scatter = Sphere.cart_scatter(10000)
        else:
            self.scatter = scatter
        
    def grad_eval(self, p): #evaluate gradient at point p=(x,y,z)

        return self.grad.subs([(x,p[0]), (y,p[1]), (z, p[2])])

    def k_nns(self, k):
        #find the k nearest neighbors to a point p, where p is assumed to be one of the points of the scatter
        #Use direct approach with sklearn library

        X = self.scatter
        nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'kd_tree').fit(X) #using kd_tree due to the low dimensionality of the problem D=3
        distances, indices = nbrs.kneighbors(X)

        return distances, indices

    def calc_normal_LMS(self, k):
        #To calculate the normal we use LMS = Least Mean Squares. The normal to p1, p2, p3,..., pk
        #is obtained by minimized e = sum(pi*n - d), n being the normal to the hyperplane n*P - d =0
        #This is solved through minimizing ||An-d|| where A=(p1,p2,...,pk)

        _, indices = self.k_nns(k) #group the scatter to k nearest neighbors
        normals = []
        v = np.zeros(3) #initializing v the normal

        for kneighbors_of_i in indices: #indices contains lists of k nearest neighbors, indices[i] = [i, *, *,..., *] with i naturally being closest to i
            arr = np.array([self.scatter[k] for k in kneighbors_of_i], dtype=float) #the k vectors closest to vector i
            v = np.linalg.lstsq(arr, np.ones((arr.shape[0], 1), dtype=arr.dtype), rcond=None)[0].ravel() #LMS by numpy
            v = v / np.linalg.norm(v) #normalize
            normals.append(v)
        
        return normals

    def step_flow(self, k=50):
        normals = self.calc_normal_LMS(k)
        v = np.zeros(3)
        gradient = np.zeros(3)
        for i in range(len(self.scatter)):
            gradient = self.grad_eval(self.scatter[i])
            v = gradient - np.inner(gradient, normals[i]) * normals[i]
            self.scatter[i] = np.array(self.scatter[i] + v, dtype = float) #forcing this to be numpy array of dtype = float
            #because otherwise i'm getting the following error:
            #----------------
            #utureWarning: Arrays of bytes/strings is being converted to decimal numbers if dtype='numeric'.
            #This behavior is deprecated in 0.24 and will be removed in 1.1 (renaming of 0.26).
            #Please convert your data to numeric values explicitly instead.
            #---------------

    
    def plot_scatter(self, projections = False): #set projections = True if you want to see projections on the cardinal axes
        #generate the unit sphere
        r = 1
        phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0:2.0 * np.pi : 100j]
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        points = self.scatter
            
        xx, yy, zz = zip(*points)

        #Set colours and render
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(
            x, y, z,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

        ax.scatter(xx,yy,zz,color="k",s=20)

        if projections == True:
            ax.plot(xx, zz, 'r+', zdir='y', zs=2)
            ax.plot(yy, zz, 'g+', zdir='x', zs=-2)
            ax.plot(xx, yy, 'b+', zdir='z', zs=-2)

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        plt.tight_layout()
        plt.show()