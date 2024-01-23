# Déclaration de recherche personnelle

# Je soussigné(e), LAMBRECHT Pierre-Antoine, étudiant(e) en Calcul Scientifique 2, déclare par la présente que le travail que je soumets, reflète une recherche personnelle effectuée dans le cadre de ce cours. Je certifie que toutes les informations, analyses, idées et conclusions présentées dans ce travail sont le résultat de mon propre effort intellectuel et de mes propres recherches.

# Je reconnais avoir respecté les principes éthiques de l'intégrité académique tout au long du processus de recherche et de rédaction de ce travail. Je m'engage à fournir ci-après toutes les sources d'information, qu'elles soient imprimées, électroniques ou orales, conformément aux normes de citation académique prévues par le cours.

# Je comprends que la falsification, le plagiat ou toute autre forme de tricherie académique sont des violations graves du code de conduite académique de notre institution et peuvent entraîner des sanctions sévères, y compris l'échec du cours.

# Je m'engage à assumer la responsabilité totale du contenu de ce travail et à accepter les conséquences de tout acte contraire à l'intégrité académique.

# Je déclare avoir utilisé les sources suivantes :
# - le cours
# - wikipedia
# - chatGPT (je ne l'utilise pas pour me faire les exo mais plutot comme un super google ou pour me debloque/debugguer
# ou parfois tout à la fin pour voir si je peux ameliorer/optimiser mon code)
# - scipy documentation
# - mon cours de TAN S5
# - networkx documentation
# - numpy documentation

import numpy as np
import networkx as nx
import scipy.linalg as sc

TOL = 1e-8

# Je pensais, d'apres les consignes, qu'il ne fallait pas du tout utilisé de list
# python, seulement des ndarray, peut être pour des raisons d'optimisation,
# les tableau dynamique c'est couteux. Du coup j'ai un peu galérer par moments
# et le code est un peu moins lisible qu'avec des 'append' sur des list.


# Exercice 1 - CSR
def exo1(M):
    """CSR storage for sparses matrices.

    param M : ndarray, matrix to store.

    return (V,C,R): 3-tuple of ndarray,
            V = nnz coeffs,
            C = column indexes of M,
            R = indexes of the first nnz in V for each row and R[-1] = nb of nnz+1.
    """
    n, m = np.shape(M)
    nb_nnz = np.count_nonzero(M)  # we need the number of nnz to init ndarrays

    if nb_nnz == 0:
        print("The matrix is the zero matrix, there is nothing to do")
        return None

    V = np.zeros(nb_nnz, dtype=float)  # 1D ndarrays
    C = np.zeros(nb_nnz, dtype=int)
    R = np.zeros(n + 1, dtype=int)
    R[-1] = nb_nnz + 1

    v_idx = 0  # ndarray cant be appended, so we keep track of indexes to use
    r_idx = 0

    for i in range(n):
        first_nnz = False  # we keep track of the first nnz per row

        for j in range(m):
            if M[i, j] != 0:
                V[v_idx] = M[i, j]  # array of nnz
                C[v_idx] = j  # array of nnz col indexes
                v_idx += 1  # found 1 nnz so we ++idx

                if first_nnz == False:
                    R[r_idx] = v_idx - 1  # index in V of the first nnz of the row
                    first_nnz = True  # we found the first nnz for this row
                    r_idx += 1
    return V, C, R


# Exercice 2 - Skyline
def exo2(M):
    """SKS storage for sparse skyline format matrices.

    param M : ndarray, matrix to store.

    return (I,V) : 2-tuple of ndarray,
                I[i] = first nnz of the i-th column (1st col doesnt count and last coeff is number of nnz+1)
                V = nnz of the upper triangular part read col by col, left to right.
    """
    n, m = np.shape(M)
    nb_nnz_total = np.count_nonzero(M)

    if nb_nnz_total == 0:
        print("The matrix is the zero matrix, there is nothing to do")
        return None

    v_idx = 1
    V = [M[0, 0]]
    I = []

    for j in range(1, m):  # column by column
        first_nnz = True  # we start with having found the 1st nnz in each column

        for i in range(j + 1):
            if M[i, j] != 0:  # we look for the 1st nnz of the column
                if first_nnz:
                    I.append(v_idx)  # store the first nnz idx of the column
                    first_nnz = False

                    for k in range(i, j + 1):
                        V.append(M[k, j])
                        v_idx += 1
    I.append(len(V) + 1)
    return np.array(I), np.array(V)


# Exercice 3 - multiplication A (Skyline) * b
def exo3(I, V, b):
    """Multiply a symmetric skyline matrix by a vector.

    param I : ndarray, from SKS decomposition,
            I[i] = first nnz of the i-th column (1st col doesnt count and last coeff is number of nnz+1)
    param V : ndarray, from SKS decomposition,
            V = nnz of the upper triangular part read col by col, left to right.
    param b: ndarray, vector of values.

    return Ab: ndarray, result of the product.
    """
    n = len(I)
    m = len(b)
    if n != m or n == 0:
        raise ValueError("Error: matrix and vector size are incompatible")

    Ab = np.zeros(n, dtype=float)

    # nb_nnz_col[i] = nb of nnz on the i-th col
    nb_nnz_col = np.zeros(n, dtype=int)

    # diag_start[i] = idx of where the diag of the i-th col starts in V
    diag_start = np.zeros(n, dtype=int)

    nb_nnz_col[0] = 1  # diag is full, A[0,0] is a nnz
    diag_start[0] = 0
    diag_start[-1] = V.shape[0] - 1  # = I[-1] - 2 also works
    for i in range(1, n - 1):
        nb_nnz_col[i] = I[i] - I[i - 1]
        diag_start[i] = I[i] - 1

    nnz = sum(nb_nnz_col)
    nb_nnz_col[-1] = I[-1] - nnz - 1
    # print(nb_nnz_col)
    # print(diag_start)

    Ab[0] += V[0] * b[0]
    for col in range(1, n):
        size_col = nb_nnz_col[col]
        for j in range(size_col):
            d = diag_start[col]
            row = col - j

            # print(f"Ab[{col}] += b[{row}] * {V[d - j]}")
            Ab[col] += b[row] * V[d - j]

            # we dont want to count the diagonal twice
            if j != 0:
                # print(f"Ab[{row}] += b[{row+j}] * {V[d - j]}")
                Ab[row] += b[row + j] * V[d - j]
    return Ab


# Exercice 4 - factorisation LU avec pivot partiel
def exo4(M):
    """PM=LU factoriation with partial pivot.

    param M : ndarray, square matrix to factorize, all main minors should be non zero,

    return (L,U,P) : 3-tuple of ndarrays,
                    L = lower triangulare matrix with diag elem = 1,
                    U = upper triangulare matrix,
                    P = permutation matrix used to keep track of the permutations.
    """
    n, m = np.shape(M)
    if n != m or n == 0:
        raise ValueError("Error: M must be a square matrix")

    L = np.zeros((n, n), dtype=float)
    P = np.eye(n)
    U = np.copy(M)

    for i in range(n):
        # we search the abs max value of the i-th col
        # argmax gives us the index of this val
        max_row = np.argmax(np.abs(U[i:, i])) + i  # +i to correct indexes in the view
        # [[]] is used to swap rows
        L[[i, max_row]] = L[[max_row, i]]
        P[[i, max_row]] = P[[max_row, i]]
        U[[i, max_row]] = U[[max_row, i]]
        L[i, i] = 1

        for k in range(i + 1, n):
            if abs(U[i, i]) < TOL:
                raise ValueError("Error: pivot is too close to zero")
            L[k, i] = U[k, i] / U[i, i]
            U[k, 0:n] = U[k, 0:n] - U[k, i] / U[i, i] * U[i, 0:n]

    return L, U, P


# Exercice 5 - résolution d'un système linéaire avec/sans pivot
def Descente(L, b):
    n = np.size(b)
    y = np.zeros(n, dtype=float)

    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        S = np.sum(L[i, 0:i] * y[0:i])
        y[i] = (b[i] - S) / L[i, i]

    return y


def Remontee(U, y):
    n = np.size(y)
    x = np.zeros(n, dtype=float)

    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        S = np.sum(U[i, i + 1 : n] * x[i + 1 : n])  # operation terme a terme
        x[i] = (y[i] - S) / U[i, i]

    return x


def exo5(L, U, b, P=None):
    """Solve linear system with/without partial pivot PAx=Pb, P is optional.

    param L : ndarray, lower triangular matrix from PA=LU factorization.
    param U : ndarray, upper triangular matrix from PA=LU factorization.
    param b : ndarray.
    param P : ndarray, optional pivot matrix from PA=LU factorization.

    return x : ndarray, solution of the system.
    """
    if P is not None:
        b = P @ b  # not in place, b is preserved
    # PA = LU
    # PAx = Pb
    y = Descente(L, b)  # we solve Ly=b
    x = Remontee(U, y)  # we solve Ux=y

    return x


# Exercice 6 - Algo de Cuthill-McKee
def exo6666(M):
    # todo : get rid of list and use ndarrays
    """CMK algorithm.

    param M : ndarray, matrix to reduce.

    return P : ndarray, permutation matrix of the reduction.
    """
    G = nx.Graph(M)
    perm = []  # permutation list

    max_ecc = -1  # initialisation of max_eccentricity
    ecc_nodes = []  # all the eccentric nodes will be put in there

    # we look for nodes on the edge of the graph
    for node in G.nodes():
        ecc = nx.eccentricity(G, v=node)
        if ecc >= max_ecc:
            max_ecc = ecc
            ecc_nodes.append(node)

    # amongs those nodes we look for the one with the highest degree -> s0
    max_deg = -1
    for node in ecc_nodes:
        deg = len(list(nx.neighbors(G, node)))
        if deg > max_deg:
            max_deg = deg
            s0 = node

    perm.append(s0)  # we start our permutation list with s0
    current = [s0]
    visited = [s0]

    for i in range(max_ecc):
        next_nodes = []
        for node in current:
            # we sweep for the node's neighbors and mark this current one as visited
            voisin = list(nx.neighbors(G, node))
            visited.append(node)

            # in its neighbors we look for unvisited nodes and put them in the next one to visit
            # and add them to the permutation list [s0, s1, ..., sn]
            for v in voisin:
                if v not in visited and v not in perm:
                    perm.append(v)
                    next_nodes.append(v)

        current = next_nodes  # we repeat the process with the next batch

    n = len(perm)
    # print("perm= ", perm)
    # we create our permutation matrix with the permutation list obtained
    P = np.zeros(shape=(n, n), dtype=int)
    for i, p in enumerate(perm):
        P[i, p] = 1

    return P


def exo6(M):
    """CMK algorithm.

    param M : ndarray, matrix to reduce.

    return P : ndarray, permutation matrix of the reduction.
    """
    G = nx.Graph(M)  # G = (V, E, phi)
    size = G.size()  # size = n = |V| = nb of nodes

    # how to find a good candidate for s0 ?
    # we chose to do it this way :
    # s0 = highest degree of all the most eccentric nodes (ie nodes on the edge of G)

    max_ecc = -1

    # store eccentricities for each node
    ecc = np.zeros(size, dtype=int)

    # calculate eccentricities for each nodes
    for i, node in enumerate(G.nodes()):
        ecc[i] = nx.eccentricity(G, v=node)
        if ecc[i] > max_ecc:
            max_ecc = ecc[i]

    # find nodes with maximum eccentricity
    ecc_nodes = np.where(ecc == max_ecc)[0]  # returns indexes

    # among those nodes, we look for the one with the highest degree -> s0
    max_deg = -1

    for node in ecc_nodes:
        deg = len(list(nx.neighbors(G, node)))
        if deg > max_deg:
            max_deg = deg
            s0 = node

    perm = np.array([s0], dtype=int)  # init permutation array with s0
    current = np.array([s0], dtype=int)  # current batch of nodes
    visited = np.array([s0], dtype=int)  # already visited nodes

    for i in range(max_ecc):
        next_nodes = np.array([], dtype=int)
        for node in current:
            # we sweep for the node's neighbors and mark this current one as visited
            voisin = list(nx.neighbors(G, node))
            visited = np.concatenate((visited, [node]), axis=0)

            # in its neighbors, we look for unvisited nodes and put them in the next one to visit
            # and add them to the permutation list [s0, s1, ..., sn]
            for v in voisin:
                if v not in visited and v not in perm:
                    perm = np.concatenate((perm, [v]), axis=0)
                    next_nodes = np.concatenate((next_nodes, [v]), axis=0)

        current = next_nodes  # we repeat the process with the next batch

    n = len(perm)
    P = np.zeros((n, n), dtype=int)
    # print("perm= ", perm)
    # we create our permutation matrix with the permutation list obtained

    for i, p in enumerate(perm):
        P[i, p] = 1

    return P


# Exercice 7 - Profile
def exo7(I, V):
    """Compute the profile of a skyline matrix decomposed like in exo2.

    param I : ndarray, from SKS decomposition,
            I[i] = first nnz of the i-th column (1st col doesnt count and last coeff is number of nnz+1)
    param V : ndarray, from SKS decomposition,
            V = nnz of the upper triangular part read col by col, left to right.

    return p : int, profile of the skyline matrix.
    """
    n = len(I)  # size of the original matrix
    p = n  # initialize the profile sum with n

    nb_nnz_col = np.zeros(n, dtype=int)
    nb_nnz_col[0] = 1  # diag is full, A[0,0] is a nnz
    for i in range(1, n - 1):
        nb_nnz_col[i] = I[i] - I[i - 1]

    nnz = sum(nb_nnz_col)
    nb_nnz_col[-1] = I[-1] - nnz - 1  # I[-1] is nb nnz +1

    for i in range(1, n):
        h = nb_nnz_col[i] - 1  # -1 because we dont count the diag
        p += i - (i - h)  # = h ...

    return p


def test():
    # ex1 : VCR => comparé à celle du cours
    # ex2 : SKS => comparé cours
    for n in range(2, 300, 10):
        A = np.zeros((n, n))
        upper = np.random.rand(n, n)
        A = upper + upper.T - np.diag(upper.diagonal())
        b = np.random.rand(n)

        # ex3 : skyline * vector product
        I, V = exo2(A)
        assert np.linalg.norm(exo3(I, V, b) - (A @ b), 2) < TOL

        # ex4 : PA = LU decomposition
        p, l, u = sc.lu(A)
        L, U, P = exo4(A)
        assert np.abs(np.linalg.det(A)) > TOL  # check if A is inv
        assert np.linalg.norm(L - l) < TOL
        assert np.linalg.norm(U - u) < TOL
        assert np.linalg.norm(L @ U - P @ A) < TOL

        # ex5 : solve Ax = b
        # * There are some precision issue esp with large matrices
        # * maybe solve from scipy is too optimised ?
        x1 = exo5(L, U, b, P)
        x2 = np.linalg.solve(A, b)
        # if np.linalg.norm(x1 - x2) > 1e-7:
        #     print("n= ", n, " norme= ", np.linalg.norm(x1 - x2))
        assert np.linalg.norm(x1 - x2) < 1e-5

        # ex6 : CMK => manual check on paper -> ok with our conditions for S0
        # ex7 => manual check, seems ok
    return 0


def main():
    A = np.array([[0, 1, 0], [1, 2, 0], [3, 0, 1]])
    V, C, R = exo1(A)
    print("Exercice 1 : ")
    print("V \n", V)
    print("C \n", C)
    print("R \n", R)

    B = np.array(
        [
            [1, 2, 4, 0, 0],
            [2, 3, 0, 7, 0],
            [4, 0, 6, 0, 10],
            [0, 7, 0, 9, 11],
            [0, 0, 10, 11, 12],
        ],
        dtype=float,
    )
    print("\nExercice 2 : ")
    I, V = exo2(B)
    print("I \n", I)
    print("V \n", V)

    print("\nExercice 3 : ")
    b = np.random.rand(5)
    Bb = exo3(I, V, b)
    print("Bb= ", Bb)
    print("Vérif: \nB@b= ", B @ b)
    # print(np.linalg.norm(Bb - (B @ b), 2))

    print("\nExercice 4 :")
    A = np.array([[2, 4, 1], [4, 9, 5], [6, 15, 11]], dtype=float)
    A = np.random.rand(3, 3)
    p, l, u = sc.lu(A)
    L, U, P = exo4(A)

    print(f"L\n{L}\n l\n{l}\n")
    print(f"U\n{U}\n u\n{u}\n")
    print(f"P\n{P}\n p\n{p}\n")

    print("A\n", A)
    print("P^-1 @ L @ U \n", np.linalg.inv(P) @ L @ U)
    print("p^-1 @ l @ u \n", np.linalg.inv(p) @ l @ u)

    print("\nExercice 5 :")
    A = np.random.rand(3, 3)
    b = np.random.rand(3)
    L, U, P = exo4(A)
    # b = np.ones(3)
    x1 = exo5(L, U, b, P)
    x2 = np.linalg.solve(A, b)

    print("x1= ", x1)
    print("Vérif sc: \nx2= ", x2)

    print("\nExercice 6 :")
    A = np.array([[1, 0, 1, 0], [0, 0, 4, 4], [1, 2, 0, 0], [0, 0, 1, 4]])
    print("A\n", A)
    print("P\n", exo6666(A))
    print("P\n", exo6(A))

    print("\nExercice 7 :")
    # A = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    # I, V = exo2(A)
    print("Profile p= ", exo7(I, V))


main()
test()
