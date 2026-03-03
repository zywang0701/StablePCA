import numpy as np

def _random_orthonormal(p, r, avoid=None, rng=None):
    """
    Generate a p x r random orthonormal matrix.
    If avoid is provided (p x s orthonormal), ensure columns are orthogonal to avoid.
    """
    if rng is None:
        rng = np.random.default_rng()
    G = rng.normal(size=(p, r))
    if avoid is not None:
        P = avoid @ avoid.T            # projector onto span(avoid)
        G = G - P @ G                  # project G to orthogonal complement
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]


def generate_stablepca_data(
    n,              # samples per source
    p,              # feature dimension
    k,              # shared latent dimension
    m,              # source-specific latent dimension
    L,              # number of source domains
    ood_factor=20,  # number of OOD domains
    n_ood=None,     # samples per OOD domain
    scale_shared=1.0,
    noise_std=0.1,
    random_state=None,
):
    """
    Realistic StablePCA data generator.
    Each source has:
      X_l = Z_l A^T * scale_shared + S_l B_l^T * t_l + ε_l

    - A is shared signal subspace (p x k).
    - B_l are random style subspaces (p x m), different per domain.
    - t_l are style scales (one domain small, others large).
    OOD domains have new random B_ood distributed similarly.
    """
    rng = np.random.default_rng(random_state)
    if n_ood is None:
        n_ood = n

    # ------------------------------------------------------
    # 1. Shared signal loading A
    # ------------------------------------------------------
    A = _random_orthonormal(p, k, avoid=None, rng=rng)

    # ------------------------------------------------------
    # 2. Source-specific random style loadings B_l
    # ------------------------------------------------------
    B_list = []
    for _ in range(L):
        B_l = _random_orthonormal(p, m, avoid=A, rng=rng)
        B_list.append(B_l)

    # ------------------------------------------------------
    # 3. Style scales t_l
    # ------------------------------------------------------
    t_list = rng.uniform(low=0.2, high=2.0, size=L)

    # ------------------------------------------------------
    # 4. Generate source data
    # ------------------------------------------------------
    X_list = []
    for l in range(L):
        Z_l = rng.normal(size=(n, k))
        S_l = rng.normal(size=(n, m))
        eps = noise_std * rng.normal(size=(n, p))

        X_l = (Z_l @ A.T) * scale_shared \
            + (S_l @ B_list[l].T) * t_list[l] \
            + eps
        X_list.append(X_l)

    # ------------------------------------------------------
    # 5. Generate OOD domains (new style directions)
    # ------------------------------------------------------
    L_ood = ood_factor * L
    B_ood_list = []
    X_ood_list = []

    for _ in range(L_ood):
        B_ood = _random_orthonormal(p, m, avoid=A, rng=rng)
        B_ood_list.append(B_ood)

        Z_j = rng.normal(size=(n_ood, k))
        S_j = rng.normal(size=(n_ood, m))

        # OOD style level also random—sometimes high, sometimes low
        t_j = rng.uniform(0.3, 2.0) #scale_spec * rng.uniform(0.3, 1.5)

        eps = noise_std * rng.normal(size=(n_ood, p))

        X_ood = (Z_j @ A.T) * scale_shared \
              + (S_j @ B_ood.T) * t_j \
              + eps
        X_ood_list.append(X_ood)

    return X_list, X_ood_list, A, B_list, B_ood_list


def generate_stablepca_data_fix(
        n,              # samples per source
        p,              # feature dimension
        k,              # shared latent dimension
        m,              # source-specific latent dimension
        L,              # number of source domains
        random_state1=None, 
        random_state2=None
    ):
    rng1 = np.random.default_rng(random_state1)
    rng2 = np.random.default_rng(random_state2)
    
    # ------------------------------------------------------
    # 1. Shared signal loading A
    # ------------------------------------------------------
    A = _random_orthonormal(p, k, avoid=None, rng=rng1)

    # ------------------------------------------------------
    # 2. Source-specific random style loadings B_l
    # ------------------------------------------------------
    B_list = []
    for _ in range(L):
        B_l = _random_orthonormal(p, m, avoid=A, rng=rng1)
        B_list.append(B_l)

    # ------------------------------------------------------
    # 3. Style scales t_l
    # ------------------------------------------------------
    t_list = rng1.uniform(low=0.2, high=2.0, size=L)

    # ------------------------------------------------------
    # 4. Generate source data
    # ------------------------------------------------------
    X_list = []
    for l in range(L):
        Z_l = rng2.normal(size=(n, k))
        S_l = rng2.normal(size=(n, m))
        eps = 0.1 * rng2.normal(size=(n, p))

        X_l = (Z_l @ A.T) * 1 \
            + (S_l @ B_list[l].T) * t_list[l] \
            + eps
        X_list.append(X_l)
    
    # ------------------------------------------------------
    # 5. Compute population covariance matrix for each group
    # ------------------------------------------------------
    Sigma_list = []
    for l in range(L):
        Sigma_l = A @ A.T + (t_list[l] ** 2) * (B_list[l] @ B_list[l].T) + 0.01 * np.eye(p)
        Sigma_list.append(Sigma_l)

    return X_list, Sigma_list, A, B_list