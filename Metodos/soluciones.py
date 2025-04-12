import numpy as np

#Metodo para resolver por Gauss_Jordan
def gauss_jordan(A, b, tol=1e-12):
    """
    Resuelve un sistema de ecuaciones lineales usando Gauss-Jordan con pivoteo parcial.

    Args:
        A: Matriz de coeficientes (n x n)
        b: Vector de términos independientes (n)
        tol: Tolerancia para detectar ceros (opcional)

    Returns:
        x: Vector solución (n)

    Raises:
        ValueError: Si la matriz es singular o el sistema no tiene solución única
    """
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])  # Matriz aumentada

    for i in range(n):
        # Pivoteo parcial: buscar la fila con mayor valor absoluto en la columna i
        max_row = np.argmax(np.abs(M[i:, i])) + i
        if abs(M[max_row, i]) < tol:
            raise ValueError("La matriz es singular o casi singular. No se puede resolver el sistema.")

        # Intercambiar filas si es necesario
        if max_row != i:
            M[[i, max_row]] = M[[max_row, i]]

        # Normalizar la fila pivote
        pivot = M[i, i]
        M[i] = M[i] / pivot

        # Eliminación en todas las demás filas
        for j in range(n):
            if i != j:
                M[j] = M[j] - M[j, i] * M[i]

    # Verificar que la solución sea válida
    if any(np.isnan(M[:, -1])) or any(np.isinf(M[:, -1])):
        raise ValueError("El sistema no tiene solución única o es inestable numéricamente.")

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

#Metodo para resolver por Cramer
def cramer(A, b):
    """
    Resuelve un sistema de ecuaciones lineales usando la regla de Cramer.

    Args:
        A: Matriz de coeficientes (n x n)
        b: Vector de términos independientes (n)

    Returns:
        x: Vector solución (n)
    """
    det_A = np.linalg.det(A)
    n = len(b)
    x = np.zeros(n)

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A

    return x

#Metodo para resolver por LU Descomposicion
def lu_decomposition(A, b, tol=1e-12):
    """
    Resuelve un sistema de ecuaciones lineales usando descomposición LU con pivoteo parcial.

    Args:
        A: Matriz de coeficientes (n x n)
        b: Vector de términos independientes (n)
        tol: Tolerancia para detectar singularidad

    Returns:
        x: Vector solución (n)

    Raises:
        ValueError: Si la matriz es singular
    """
    n = len(b)
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.eye(n)  # Matriz de permutación
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(np.abs(A[i:, i])) + i
        pivot = A[max_row, i]

        if abs(pivot) < tol:
            raise ValueError("La matriz es singular o casi singular. No se puede resolver el sistema.")

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]

        # Descomposición LU
        U[i, i:] = A[i, i:] - np.dot(L[i, :i], U[:i, i:])
        L[(i + 1):, i] = (A[(i + 1):, i] - np.dot(L[(i + 1):, :i], U[:i, i])) / U[i, i]

    # Sustitución hacia adelante (Ly = Pb)
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.dot(P, b)[i] - np.dot(L[i, :i], y[:i])

    # Sustitución hacia atrás (Ux = y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < tol:  # Verificación adicional
            raise ValueError("División por cero detectada. La matriz es singular.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x





