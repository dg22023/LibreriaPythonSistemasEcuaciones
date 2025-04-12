from Metodos.Gauss_seidel import *

A = np.array([[3,-0.1,-0.2],[0.1,7,-0.3],[0.3,-0.2,10]], dtype=float)
b = np.array([7.85,-19.3,71.4], dtype=float)

ejemplo_gauss_seidel = guass_seidel(A, b)
print(f"Solucion por gauss seidel: {ejemplo_gauss_seidel}")

# Mostrar resultados
print(f"\nRespuestas Metodo Seidel:")
print(f"x: {ejemplo_gauss_seidel[0]:.2f}%")
print(f"y: {ejemplo_gauss_seidel[1]:.2f}%")
print(f"z: {ejemplo_gauss_seidel[2]:.2f}%")