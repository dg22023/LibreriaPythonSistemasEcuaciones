import numpy as np

#Metodo para resolver por Gauss_Jordan
def gauss_jordan(A, b):
    """
      Resuelve un sistema de ecuaciones lineales usando Gauss-Jordan.

      Args:
          A: Matriz de coeficientes (n x n)
          b: Vector de términos independientes (n)

      Returns:
          x: Vector solución (n)
      """
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])

    for i in range(n):
        M[i] = M[i] / M[i, i]
        for j in range(n):
            if i != j:
                M[j] = M[j] - M[j, i] * M[i]
    return M[:, -1]

#Metodo para resolver por Gauss_Elimination
def gaussian_elimination(A, b):
    """
    Resuelve un sistema de ecuaciones lineales usando eliminación gaussiana.

    Args:
        A: Matriz de coeficientes (n x n)
        b: Vector de términos independientes (n)

    Returns:
        x: Vector solución (n)
    """
    n = len(b)
    ab = np.hstack([A, b.reshape(-1, 1)])  # Matriz aumentada

    # Eliminación hacia adelante
    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(np.abs(ab[i:, i])) + i
        ab[[i, max_row]] = ab[[max_row, i]]

        # Eliminación
        for j in range(i + 1, n):
            factor = ab[j, i] / ab[i, i]
            ab[j, i:] -= factor * ab[i, i:]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (ab[i, -1] - np.dot(ab[i, i + 1:n], x[i + 1:n])) / ab[i, i]

    return x




