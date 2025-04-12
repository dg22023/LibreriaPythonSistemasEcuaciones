from Metodos.Jacobi import *

A = np.array([[3,-0.1,-0.2],[0.1,7,-0.3],[0.3,-0.2,10]], dtype=float)
b = np.array([7.85,-19.3,71.4], dtype=float)


ejemplo_jacobi =  jacobi(A,b)
print(f"Solucion por gauss seidel: {ejemplo_jacobi}")