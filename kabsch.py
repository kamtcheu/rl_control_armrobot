import numpy as np


import numpy as np

def kabsch(P_sim, P_real):
    """
    P_sim  : tableau (3, 3) des points MuJoCo
    P_real : tableau (3, 3) des points UR3e

    Retourne R (3x3), t (3,)
    tels que P_real ≈ R @ P_sim + t
    """

    P_sim = np.asarray(P_sim, dtype=float)
    P_real = np.asarray(P_real, dtype=float)

    # 1. Calcul des centres
    c_sim = np.mean(P_sim, axis=0)
    c_real = np.mean(P_real, axis=0)

    # 2. Centrage des points
    A = P_sim - c_sim
    B = P_real - c_real

    # 3. Matrice de covariance
    H = A.T @ B

    # 4. Décomposition SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Rotation (sans réflexion)
    D = np.eye(3)
    D[-1, -1] = np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ D @ U.T

    # 6. Translation
    t = c_real - R @ c_sim

    return R, t


def trouver_transformation_rigide(P, Q):
    """Calcule la rotation R (3x3) et la translation t (3,) optimales

    pour aligner l'ensemble de points P sur l'ensemble Q (Q ≈ R @ P + t).

    Paramètres:
        P : array numpy de taille (N, 3) - points de départ
        Q : array numpy de taille (N, 3) - points d'arrivée correspondants

    Retourne:
        R : matrice de rotation (3, 3)
        t : vecteur de translation (3,)
        rmse : erreur quadratique moyenne après alignement
    """
    assert P.shape == Q.shape, "Les deux ensembles doivent avoir la même taille."
    assert P.shape[0] >= 3, "Il faut au moins 3 couples de points."

    # 1. Calcul des centres de gravité (centroïdes)
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # 2. Centrage des nuages de points
    P_centre = P - centroid_P
    Q_centre = Q - centroid_Q

    # 3. Matrice de covariance et SVD
    H = P_centre.T @ Q_centre
    U, S, Vt = np.linalg.svd(H)

    # Calcul de la matrice de rotation R
    R = Vt.T @ U.T

    # Correction pour éviter une réflexion (effet miroir)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 4. Calcul du vecteur de translation t
    t = centroid_Q - R @ centroid_P

    # Calcul de l'erreur d'alignement (RMSE) pour vérification
    P_aligne = (R @ P.T).T + t
    rmse = np.sqrt(np.mean(np.sum((Q - P_aligne) ** 2, axis=1)))

    return R, t, rmse


# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
if __name__ == "__main__":
    # Vos points de départ (N lignes, 3 colonnes : X, Y, Z)
    points_depart = np.array(
        [[-457.34693241, -510.86994844,   66.00817931 ], [-363.10530948, -582.6533083,   181.2470076 ], [262.16977537, -293.09953005,   79.98498175]]
    )

    # Vos points d'arrivée connus (ex: rotation de 90° autour de Z + translation)
    points_arrivee = np.array(
        [[-457.08, -483.02, 66.75], [-367.46, -555.09, 181.83], [267.89, -273.62, 89.18]]
    )

    # Calcul de la transformation
    R, t, erreur = trouver_transformation_rigide(points_depart, points_arrivee)

    print("Matrice de rotation R :\n", np.round(R, 4))
    print("\nVecteur de translation t :\n", np.round(t, 4))
    print(f"\nErreur moyenne (RMSE) : {erreur:.6f}")

    # Appliquer la transformation à un nouveau point quelconque
    nouveau_point = np.array([2.0, 3.0, 4.0])
    point_transforme = R @ nouveau_point + t
    print("\nNouveau point transformé :", np.round(point_transforme, 4))

    print("==============Kabsch 2==============")

    
    # ----------- EXEMPLE -----------

    # 3 points de MuJoCo (en mètres)
    P_sim = np.array([
        [0.30, 0.10, 0.20],
        [0.35, 0.15, 0.25],
        [0.40, 0.10, 0.30]
    ])

    # 3 points correspondants du UR3e
    P_real = np.array([
        [0.310, 0.105, 0.200],
        [0.360, 0.155, 0.250],
        [0.410, 0.105, 0.300]
    ])

    R, t = kabsch(points_depart, points_arrivee)

    print("Rotation R :\n", R)
    print("Translation t (m) :", t)
    print("Translation t (mm) :", t * 1000)

    # Application de la transformation
    P_corrige = (R @ P_sim.T).T + t

    # Vérification de l'erreur
    erreurs = np.linalg.norm(P_corrige - P_real, axis=1)

    print("Erreurs (mm) :", erreurs * 1000)
    print("Erreur moyenne (mm) :", np.mean(erreurs) * 1000)
    print("Erreur maximale (mm) :", np.max(erreurs) * 1000)
