"""Déclaration de recherche personnelle.

 Je soussigné(e), Pierre-Antoine SENGER/LAMBRECHT, étudiant(e) en Calcul Scientifique 2, déclare par
la présente que le travail que je soumets, reflète une recherche personnelle
effectuée dans le cadre de ce cours. Je certifie que toutes les informations,
analyses, idées et conclusions présentées dans ce travail sont le résultat de mon
propre effort intellectuel et de mes propres recherches.

 Je reconnais avoir respecté les principes éthiques de l'intégrité académique
tout au long du processus de recherche et de rédaction de ce travail. Je m'engage
à fournir ci-après toutes les sources d'information, qu'elles soient imprimées,
électroniques ou orales, conformément aux normes de citation académique prévues
par le cours.

 Je comprends que la falsification, le plagiat ou toute autre forme de tricherie
académique sont des violations graves du code de conduite académique de notre
institution et peuvent entraîner des sanctions sévères, y compris l'échec du
cours.

 Je m'engage à assumer la responsabilité totale du contenu de ce travail et à
accepter les conséquences de tout acte contraire à l'intégrité académique.

 Je déclare avoir utilisé les sources suivantes :
https://mathsfromnothing.au/hessenberg-decomposition/?i=1
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hessenberg.html
https://webhome.auburn.edu/~tamtiny/lecture%2010.pdf
https://dspace.mit.edu/bitstream/handle/1721.1/75282/18-335j-fall-2006/contents/lecture-notes/lec14.pdf
https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition  
scipy doc  
https://people.sc.fsu.edu/~jburkardt/classes/math3040_2014/FEM1D.pdf  
https://hplgit.github.io/INF5620/doc/pub/H14/fem/html/._main_fem-solarized005.html 
Discussion avec Antoine
ChatGTP as superGoogle 
ChatGPT with the prompts:  
-What do you think of this code? can you guess what it does? does it looks correct? *code*
-I have this code *code* but i get these errors : *errors*
"""

import numpy as np
from scipy.linalg import cholesky, eig, lapack, eigh, qr, lu


def exo1(A, method="householder"):
    """Transform a squared symmetric matrix into a tridiagonal Hessenberg matrix.

    Args:
        A (ndarray): square symmetric matrix
        method (str, optional): {givens, householder}. Defaults to 'householder'.

    Raises:
        ValueError: if matrix isn't squared or method is unknown

    Returns:
        ndarray: a tridiagonal Hessenberg matrix
    """
    # In case A is not a ndarray
    A = np.asarray(A, dtype=float)

    # Check if A is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = A.shape[0]

    # Initialize Hessenberg matrix
    H = A.copy()

    # Computes H using Givens rotation
    if method == "givens":
        # Iterate over columns to introduce zeros
        for i in range(n - 1):
            # Iterate over rows below the diagonal
            for j in range(i + 2, n):
                # Compute parameters for Givens rotation
                a = H[i + 1, i]
                b = H[j, i]
                r = np.sqrt(a**2 + b**2)
                c = a / r
                s = b / r

                # Construct the Givens rotation matrix G
                G = np.eye(n, dtype=float)
                G[i + 1, i + 1] = c
                G[j, j] = c
                G[i + 1, j] = s
                G[j, i + 1] = -s

                # Apply Givens rotation to update H
                H = G @ H @ G.T

    # Compute H using Householder reflection
    elif method == "householder":
        # Iterate over subdiagonals
        for k in range(n - 2):
            # Extract the subvector x from the subdiagonal
            x = H[k + 1 :, k]

            # Initialize the Householder vector v with zeros
            v = np.zeros_like(x)

            # Compute the first element of v
            v[0] = np.sign(x[0]) * np.linalg.norm(x) + x[0]

            # Copy the remaining elements of x to v
            v[1:] = x[1:]

            # Normalize v to have unit length
            v /= np.linalg.norm(v)

            # Apply Householder reflection to update H
            H[k + 1 :, k:] -= 2.0 * np.outer(v, np.dot(v, H[k + 1 :, k:]))
            H[:, k + 1 :] -= 2.0 * np.outer(np.dot(H[:, k + 1 :], v), v)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'givens' or 'householder'.")

    return H


def exo2(n=100):
    """Estimate the coercivity constant of a given problem using finite element discretization.

    Args:
        n (int, optional): Number of discretization points. Defaults to 100.

    Returns:
        float: Approximate coercivity constant.
    """
    # I=(0,1), u(0)=u(1)=0 Dirichlet
    # Step size
    h = 1.0 / (n - 1)

    # Construct the matrices for the generalized eigenvalue problem
    Ah = 2.0 * np.eye(n, dtype=float)  # Main diag
    Ah += np.diag(-1 * np.ones(n - 1), k=-1)  # Lower diag
    Ah += np.diag(-1 * np.ones(n - 1), k=1)  # Upper diag
    Ah *= 1.0 / h

    Mh = 4.0 * np.eye(n, dtype=float)  # Main diag
    Mh += np.diag(np.ones(n - 1), k=-1)  # Lower diag
    Mh += np.diag(np.ones(n - 1), k=1)  # Upper diag
    Mh *= h / 6.0

    # Cholesky decomposition, Mh = L*L.T
    L = cholesky(Mh, lower=True)
    L_inv = lapack.dtrtri(L, lower=True)[0]  # Better than solve(L, Id) ?

    # Matrix of the classical eigvalues problem
    C = L_inv @ Ah @ L_inv.T
    # print(C) # not triband

    # Calculate the eigenvalues of C
    eigenvalues, _ = eig(C)  # Uses spectral approch
    smallest_eigenvalue = min(eigenvalues)

    # Or use QR method accelarated with Hessenberg matrix from exo1 (slower)
    # iterations = 1000
    # Ak = exo1(C) # Hessenberg form
    # for _ in range(iterations):
    #     Q,R = qr(Ak)
    #     Ak = R @ Q

    # smallest_eigenvalue = min(np.diag(Ak))

    # lambda_1 = 1/m^2 <=> m =
    m = 1.0 / np.sqrt(smallest_eigenvalue)
    # print(m)
    m = np.real(m)  # To avoid warning in next integer cast

    # Truncate instead of rounding up to get true value of Cp to 2 decimal places
    # Cp = np.round(m, 2)
    Cp = int(m * 10**2) / 10**2

    return Cp


def exo3(k0, k1, n=100):
    """Solve a particular PDE using Galeskin Finite Element method.

    Args:
        k0 (float): A stricly positive parameter of the problem.
        k1 (float): A stricly positive parameter of the problem.
        n (int, optional): Number of points in the discretization. Defaults to 100.

    Raises:
        ValueError: If k0 or k1 is <=0.

    Returns:
        ndarray: Solution of the problem.
    """
    if k0 <= 0 or k1 <= 0:
        raise ValueError(f"k0 and k1 must be strictly positive (given: {k0} and {k1})")

    # I=(0,1), u(0)=u(1)=0 Dirichlet
    # Step size
    h = 1.0 / (n - 1)
    x = np.arange(0, 1, h)

    # Contruct the matrix of the problem
    # Ah = A0 + A1
    A0 = 2.0 * np.eye(n, dtype=float)  # Main diag
    A0 += np.diag(-1 * np.ones(n - 1), k=-1)  # Lower diag
    A0 += np.diag(-1 * np.ones(n - 1), k=1)  # Upper diag
    A0 *= k0 / h

    # For A1 :
    diag = np.zeros(n, dtype=float)
    diag[0] = x[1] ** 2
    for i in range(1, n - 2):
        diag[i] = x[i + 1] ** 2 - x[i - 1] ** 2
    diag[n - 2] = -x[n - 3] ** 2
    diag *= k1 / (2 * h**2)

    lower_upper = np.zeros(n - 1, dtype=float)
    for i in range(n - 2):
        lower_upper[i] = x[i + 1] ** 2 - x[i] ** 2
    lower_upper[n - 2] = -x[n - 2] ** 2
    lower_upper *= k1 / h**2

    A1 = np.diag(diag)
    A1 += np.diag(-lower_upper, k=-1)
    A1 += np.diag(-lower_upper, k=1)

    # Final matrix of the problem
    Ah = A0 + A1  # Tridiag

    # Construct the Right Hand Side Bh
    Bh = np.zeros(n)
    Bh[0] = 2 * x[0] ** 2 - x[1] ** 2
    for i in range(1, n - 2):
        Bh[i] = 2 * x[i] ** 2 - x[i - 1] ** 2 - x[i + 1] ** 2
    Bh[n - 2] = 2 * x[n - 2] ** 2 - x[n - 3] ** 2

    Bh /= 2 * h
    Bh += 1

    # Solves : Ah @ uh = Bh
    # uh = np.linalg.solve(Ah, Bh)

    # Since Ah is tridiag it should be faster with a LU factorization
    P, L, U = lu(Ah)

    # Solve Ly = Bh
    y = np.linalg.solve(L, Bh)

    # Solve Ux = y
    uh = np.linalg.solve(U, y)

    return uh
