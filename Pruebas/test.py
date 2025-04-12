from Metodos.soluciones import *

#Ejemplo Gauss_Jordan
A = np.array([
    #Cambiar los valores de acuerdo a la matriz [nxn]
    #Para una matriz [0.3,0.2,-0.3],[-0.3,0.1,0.1],[-1,1,0]
    #Sustituir el array de la siguiente forma:
    [0.3, 0.2, -0.3], [-0.3, 0.1, 0.1], [-1, 1, 0]], dtype=float)

    #Valores independientes de las ecuaciones:
b = np.array([0, 0, 100], dtype=float)

solucion = gauss_jordan(A, b)
print(f"Solución por Gauss-Jordan: {solucion}")

print(f"X: {solucion[1]:.0f}")
print(f"Y: {solucion[0]:.0f}")
print(f"Z: {solucion[2]:.0f}")

#Ejemplo Gauss_Elimination
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    x = gaussian_elimination(A, b)
    expected = np.array([2, 3, -1])
    assert np.allclose(x, expected, atol=1e-6)

#Ejemplo Gauss_Seidel

