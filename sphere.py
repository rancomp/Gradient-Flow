import numpy as np
import scipy as sp
import math
from sympy.tensor.array import derive_by_array #Symbolic gradient calculation
from sympy.abc import x,y,z #Defining symbolic parameters for sympy and for the function so that I can calc the gradient at any point
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits import mplot3d

class Sphere: #Sphere library
    

    def polar_scatter(M = 1000): #return scatter of M points on the sphere, Polar uniformly spaced (NOT RANDOM)
        sqrtm = math.isqrt(M) #integer part of the root
        theta = np.linspace(0, 2* np.pi, sqrtm) #theta coordinate
        phi = np.linspace(0, np.pi, sqrtm+1) #phi coordinate

        #need to fix to create a scatter!!!!
        return theta, phi
        #return Sphere.eval_sphere(theta[:,None], phi[None,:])
    

    def eval_sphere_polar(theta, phi):
        return [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    
    def cart_scatter(M = 10000):
        #return a some-what even scatter of M points on the sphere, Cartesian coords, uniformly spaced (NOT RANDOM)
        #using the so called Sun-flower algorithm, or Fibonacci algorithm
        #Personally I should have preferred using Coulomb potential, but couldn't be bothered with minimizing potentials.
        indices = np.arange(0, M, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices / M)
        theta = np.pi * (1 + 5**0.5) * indices

        xx, yy, zz = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        points = list(zip(xx,yy,zz))
        points = [np.array(p) for p in points] #turning the points into numpy arrays for easier matrix manipulations later

        #zip the lists together to form set of points [x,y,z] then also turn
        #them into list for ease of use (don't want to mess with iterables)
        return points
    
    def cart_scatter_rand(M = 10000):
        #Just the standard uniform distribution on the sphere (pulled-back from [0,1])
        points = []
        pi = np.pi
        for i in range(M):
            theta = 2 * pi * np.random.random_sample() #uniform random between [0,2pi]
            zz = np.random.random_sample() * 2 - 1 #uniform random between [-1,1]
            xx = np.sqrt(1-zz*zz) * np.cos(theta)
            yy = np.sqrt(1-zz*zz) * np.sin(theta)
            points.append(np.array([xx,yy,zz])) #easier to work with numpy arrays later when performing matrix calculations
        
        return points