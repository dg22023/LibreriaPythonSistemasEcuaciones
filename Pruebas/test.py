from Metodos.soluciones import *

#Ejemplo Gauss_Jordan
print("----------------------------------------")
print("Gauss_Jordan")
class TestGaussJordan:
    @staticmethod
    def test_1():
        """Sistema básico 2x2"""
        A = np.array([[4, 3],
                      [3, 2]])
        b = np.array([10, 7])

        expected = np.array([1, 2])
        result = gauss_jordan(A, b)

        print("Test 1 - Sistema básico 2x2:")
        print("Resultado:", result)
        print("¿Coincide?", np.allclose(result, expected))
        print()

    @staticmethod
    def test_2():
        """Sistema que requiere pivoteo"""
        A = np.array([[0, 2, 4],
                      [1, 1, 2],
                      [0, 1, 3]])
        b = np.array([6, 4, 4])

        expected = np.array([1, 1, 1])
        result = gauss_jordan(A, b)

        print("Test 2 - Sistema con pivoteo necesario:")
        print("Resultado:", result)
        print("¿Coincide?", np.allclose(result, expected))
        print()

    @staticmethod
    def test_3():
        """Sistema singular (debería fallar)"""
        A = np.array([[1, 2],
                      [2, 4]])  # Segunda fila es múltiplo de la primera
        b = np.array([5, 10])

        try:
            result = gauss_jordan(A, b)
            print("Test 3 - Sistema singular:")
            print("ERROR: No detectó la matriz singular")
        except ValueError as e:
            print("Test 3 - Sistema singular:")
            print("CORRECTO: Detectó la matriz singular")
            print("Mensaje de error:", str(e))
        print()

    @staticmethod
    def test_4():
        """Sistema mal condicionado"""
        A = np.array([[1, 1],
                      [1, 1.0001]])
        b = np.array([2, 2.0001])

        expected = np.array([1., 1.])
        result = gauss_jordan(A, b)

        print("Test 4 - Sistema mal condicionado:")
        print("Resultado:", result)
        print("¿Coincide?", np.allclose(result, expected, atol=1e-5))
        print()


if __name__ == "__main__":
    TestGaussJordan.test_1()
    TestGaussJordan.test_2()
    TestGaussJordan.test_3()
    TestGaussJordan.test_4()

print("----------------------------------------")

#Ejemplo Gauss_Elimination
print("Gauss_Elimination")
class TestGaussianElimination:
    @staticmethod
    def test_1():
        """Sistema con solución única"""
        A = np.array([[2, 1, -1],
                      [-3, -1, 2],
                      [-2, 1, 2]], dtype=float)
        b = np.array([8, -11, -3], dtype=float)

        expected = np.array([2, 3, -1])  # Solución conocida
        result = gaussian_elimination(A, b)

        print("Test 1 - Sistema con solución única:")
        print("Matriz A:\n", A)
        print("Vector b:", b)
        print("Solución esperada:", expected)
        print("Solución obtenida:", result)
        print("¿Coinciden?", np.allclose(result, expected))
        print()

    @staticmethod
    def test_2():
        """Sistema 2x2 simple"""
        A = np.array([[4, 3],
                      [3, 2]], dtype=float)
        b = np.array([10, 7], dtype=float)

        expected = np.array([1, 2])  # Solución conocida
        result = gaussian_elimination(A, b)

        print("Test 2 - Sistema 2x2 simple:")
        print("Matriz A:\n", A)
        print("Vector b:", b)
        print("Solución esperada:", expected)
        print("Solución obtenida:", result)
        print("¿Coinciden?", np.allclose(result, expected))
        print()

    @staticmethod
    def test_3():
        """Sistema que requiere pivoteo"""
        A = np.array([[0, 2, 0],
                      [1, 1, 0],
                      [0, 0, 3]], dtype=float)
        b = np.array([4, 3, 6], dtype=float)

        expected = np.array([1, 2, 2])  # Solución conocida
        result = gaussian_elimination(A, b)

        print("Test 3 - Sistema que requiere pivoteo:")
        print("Matriz A:\n", A)
        print("Vector b:", b)
        print("Solución esperada:", expected)
        print("Solución obtenida:", result)
        print("¿Coinciden?", np.allclose(result, expected))
        print()


# Ejecutar los tests
if __name__ == "__main__":
    TestGaussianElimination.test_1()
    TestGaussianElimination.test_2()
    TestGaussianElimination.test_3()


print("----------------------------------------")
#Ejemplo Cramer
print("Cramer")

class TestCramerRule:
    @staticmethod
    def test_1():
        """Sistema 2x2 simple"""
        A = np.array([[2, 1],
                      [1, 3]], dtype=float)
        b = np.array([5, 10], dtype=float)

        expected = np.array([1, 3])  # Solución conocida
        result = cramer(A, b)

        print("Test 1 - Sistema 2x2 simple:")
        print("Matriz A:\n", A)
        print("Vector b:", b)
        print("Solución esperada:", expected)
        print("Solución obtenida:", result)
        print("¿Coinciden?", np.allclose(result, expected))
        print()

    @staticmethod
    def test_2():
        """Sistema 3x3 con solución única"""
        A = np.array([[3, 2, -1],
                      [2, -2, 4],
                      [-1, 0.5, -1]], dtype=float)
        b = np.array([1, -2, 0], dtype=float)

        expected = np.array([1, -2, -2])  # Solución conocida
        result = cramer(A, b)

        print("Test 2 - Sistema 3x3:")
        print("Matriz A:\n", A)
        print("Vector b:", b)
        print("Solución esperada:", expected)
        print("Solución obtenida:", result)
        print("¿Coinciden?", np.allclose(result, expected))
        print()

    @staticmethod
    def test_3():
        """Sistema con solución no entera"""
        A = np.array([[1, 1],
                      [3, -2]], dtype=float)
        b = np.array([5, 0], dtype=float)

        expected = np.array([2., 3.])  # Solución conocida
        result = cramer(A, b)

        print("Test 3 - Sistema con solución fraccionaria:")
        print("Matriz A:\n", A)
        print("Vector b:", b)
        print("Solución esperada:", expected)
        print("Solución obtenida:", result)
        print("¿Coinciden?", np.allclose(result, expected))
        print()


# Ejecutar los tests
if __name__ == "__main__":
    TestCramerRule.test_1()
    TestCramerRule.test_2()
    TestCramerRule.test_3()

print("----------------------------------------")

#Ejemplo LU
print("LU")


def test_lu_decomposition():
    """Función para probar la descomposición LU con pivoteo"""

    print("=== TEST 1: Sistema 3x3 simple ===")
    A1 = np.array([[2, -1, -2],
                   [-4, 6, 3],
                   [-4, -2, 8]])
    b1 = np.array([-1, 13, -6])
    try:
        x1 = lu_decomposition(A1, b1)
        print("Solución:", x1)
        print("Verificación (A@x - b):", np.linalg.norm(A1 @ x1 - b1))
    except ValueError as e:
        print("Error:", e)

    print("\n=== TEST 2: Requiere pivoteo ===")
    A2 = np.array([[0, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    b2 = np.array([5, 15, 24])
    try:
        x2 = lu_decomposition(A2, b2)
        print("Solución:", x2)
        print("Verificación (A@x - b):", np.linalg.norm(A2 @ x2 - b2))
    except ValueError as e:
        print("Error:", e)

    print("\n=== TEST 3: Matriz singular ===")
    A3 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # Matriz singular (det = 0)
    b3 = np.array([6, 15, 24])

    try:
        x3 = lu_decomposition(A3, b3)
        print("Solución:", x3)
    except ValueError as e:
        print("CORRECTO: Sistema detectado como singular")
        print("Mensaje de error:", str(e))


if __name__ == "__main__":
    test_lu_decomposition()



