import numpy as np

def guass_seidel(A, b, tol=1e-10, max_iter=1000):

    n = len(b)
    x = np.zeros(n)

    for k in range(max_iter):
        x_prev = x.copy()

        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x - x_prev) < tol:
            print(f"Numero total de iteraciones necesitadas: {k}")
            return x

    print(f"No se pudo converger en {max_iter} iteraciones.")
    return x