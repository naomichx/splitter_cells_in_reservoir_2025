"""
Subspace alignment analysis for the splitter-cell lesion series.

Compares task-relevant (potent) and null subspaces before and after each
lesion iteration, following the Churchland-Shenoy null-space reorganisation
framework.

potent subspace  :  task axis from CCA between LR and RL crossing activations
                    (cca_core; replaces μ_LR − μ_RL)
null subspace    :  orthogonal complement of the potent direction
representation similarity : SVCCA + CCA / PWCCA / PLS (cca_core, pwcca, numpy_pls)
common neurons   :  neurons that survive all lesions up to iteration k
                    (always a subset of the original 5000; killed neurons are
                    zeroed in the weight matrices but stay in the arrays)
"""

import os
import re
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.linalg import svd as scipy_svd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.cross_decomposition import CCA
import cca_core
import numpy_pls
import pwcca

# ── import helpers from single_cell_analysis ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "sca", os.path.join(os.path.dirname(__file__), "single_cell_analysis.py"))
_sca = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sca)
load_positions          = _sca.load_positions
load_reservoir_states   = _sca.load_reservoir_states
find_location_indexes_xy = _sca.find_location_indexes_xy
find_activity_ranges    = _sca.find_activity_ranges
get_average_activity    = _sca.get_average_activity

# ── colours ───────────────────────────────────────────────────────────────────
BLUE   = '#2E86AB'
ORANGE = '#E07A5F'
GREEN  = '#3BB273'
PURPLE = '#7B2D8B'
GREY   = '#B0B0B0'
N_TOTAL = 3000   # total reservoir size (never changes)


# =============================================================================
#  1. DATA LOADING
# =============================================================================

def _iter_from_subdir(name):
    """Return iteration integer from a reservoir-states subfolder name."""
    if name == 'reservoir_states':
        return 0
    m = re.match(r'reservoir_states_1_killed_(\d+)$', name)
    return int(m.group(1)) if m else None


def load_splitter_indices(root, seed):
    """
    Return {iteration: array_of_killed_indices} for one seed.
    Indices are always in the original [0, N_TOTAL) space.
    """
    seed_path = os.path.join(root, f'seed_{seed}')
    result = {}
    for sub in os.listdir(seed_path):
        f = os.path.join(seed_path, sub, 'splitter_cells_index.npy')
        if not os.path.exists(f):
            continue
        it = _iter_from_subdir(sub)
        if it is None:
            continue
        result[it] = np.load(f, allow_pickle=True).astype(int)
    return result


def get_cumulative_killed(sc_dict):
    """
    Given sc_dict = {iter: killed_indices}, return
    {iter: boolean_mask_of_ACTIVE_neurons (True = alive)}.
    At iter k, the killed neurons are those found at iters 0..k-1.
    """
    active_masks = {}
    killed_so_far = np.zeros(N_TOTAL, dtype=bool)  # True = killed
    for it in sorted(sc_dict):
        active_masks[it] = ~killed_so_far.copy()    # alive before this lesion
        killed_so_far[sc_dict[it]] = True
    return active_masks


def build_condition_matrix(path, max_per_type=16):
    """
    Build the condition matrix for one reservoir-states folder.

    One row per LR/RL corridor crossing (mean activity over the crossing window).

    Returns
    -------
    X : ndarray, shape (n_samples, N_TOTAL)
    y : ndarray, shape (n_samples,)  — +1 LR, -1 RL
    n_lr, n_rl : int
        Number of LR / RL crossings used (for the ≥min_crossings validity gate).
    """
    res = load_reservoir_states(path)       # (T, N_TOTAL)
    pos = load_positions(path)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    loc, loc_idx = find_location_indexes_xy(x_pos, y_pos)
    ar = find_activity_ranges(loc, loc_idx)

    rows, labels = [], []
    n_lr_cross, n_rl_cross = 0, 0

    for traj, label in [('LR', +1), ('RL', -1)]:
        ranges = ar[traj]
        n_cross = len(ranges) if max_per_type is None else min(max_per_type, len(ranges))
        if traj == 'LR':
            n_lr_cross = n_cross
        else:
            n_rl_cross = n_cross

        for idx in range(n_cross):
            start, end = int(ranges[idx][0]), int(ranges[idx][1])
            end = min(end, len(res))
            start = max(0, start)
            rows.append(get_average_activity((start, end), res))
            labels.append(label)

    if not rows:
        return np.empty((0, N_TOTAL)), np.array([], dtype=float), n_lr_cross, n_rl_cross

    X = np.vstack(rows)
    y = np.array(labels, dtype=float)
    return X, y, n_lr_cross, n_rl_cross


def _match_sample_rows(X0, Xk, seed=0):
    """Subsample both matrices to the same number of rows (for SVCCA / CCA)."""
    n = min(len(X0), len(Xk))
    if n == 0:
        return X0, Xk
    if len(X0) == n and len(Xk) == n:
        return X0, Xk
    rng = np.random.default_rng(seed)
    if len(X0) > n:
        X0 = X0[rng.choice(len(X0), n, replace=False)]
    if len(Xk) > n:
        Xk = Xk[rng.choice(len(Xk), n, replace=False)]
    return X0, Xk


def _condition_to_acts(X, active_mask):
    """Centered activations (n_neurons, n_datapoints) for cca_core / pls / pca."""
    Xs = X[:, active_mask]
    acts = Xs.T.astype(np.float64)
    acts -= acts.mean(axis=1, keepdims=True)
    return acts


def _svcca_reduce_acts(cacts, n_dims):
    """SVCCA: SVD in neuron space, keep top n_dims singular directions."""
    _, s, V = np.linalg.svd(cacts, full_matrices=False)
    k = int(min(n_dims, len(s), cacts.shape[1] - 1))
    k = max(1, k)
    return s[:k, np.newaxis] * V[:k, :], k


def _effective_sv_dims(n_neurons, n_datapoints, n_sv_dims):
    """k such that reduced matrices satisfy neurons < datapoints."""
    return int(max(1, min(n_sv_dims, n_neurons, n_datapoints - 1)))


def _cca_weights_to_neurons(cacts, w_reduced):
    """Map CCA direction in SVCCA-reduced space back to neuron space."""
    w_reduced = np.asarray(w_reduced).ravel()
    U, _, _ = np.linalg.svd(cacts, full_matrices=False)
    k = min(len(w_reduced), U.shape[1])
    w = U[:, :k] @ w_reduced[:k]
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return np.zeros(cacts.shape[0])
    return (w / norm).astype(float)


# =============================================================================
#  2. ANALYSIS PRIMITIVES
# =============================================================================

def _balanced_lr_rl_acts(X, y, active_mask, seed=0):
    """Centered LR / RL activation matrices with matched column counts."""
    lr_idx = y == +1
    rl_idx = y == -1
    if lr_idx.sum() < 2 or rl_idx.sum() < 2:
        return None, None
    acts_lr = _condition_to_acts(X[lr_idx], active_mask)
    acts_rl = _condition_to_acts(X[rl_idx], active_mask)
    n = min(acts_lr.shape[1], acts_rl.shape[1])
    rng = np.random.default_rng(seed)
    if acts_lr.shape[1] > n:
        acts_lr = acts_lr[:, rng.choice(acts_lr.shape[1], n, replace=False)]
    if acts_rl.shape[1] > n:
        acts_rl = acts_rl[:, rng.choice(acts_rl.shape[1], n, replace=False)]
    return acts_lr, acts_rl


def task_potent_direction(X, y, active_mask, n_sv_dims=20, epsilon=1e-10,
                          seed=0):
    """
    Task (potent) axis via CCA between LR and RL crossing activations.

    Uses ``cca_core`` on SVCCA-reduced LR/RL matrices; returns the first
    canonical direction in neuron space (LR side).
    """
    n_active = int(active_mask.sum())
    acts_lr, acts_rl = _balanced_lr_rl_acts(X, y, active_mask, seed=seed)
    if acts_lr is None:
        return np.zeros(n_active)

    n = acts_lr.shape[1]
    k = _effective_sv_dims(acts_lr.shape[0], n, n_sv_dims)
    sv_lr, _ = _svcca_reduce_acts(acts_lr, k)
    sv_rl, _ = _svcca_reduce_acts(acts_rl, k)
    k = min(sv_lr.shape[0], sv_rl.shape[0])
    if k < 1:
        return np.zeros(n_active)

    try:
        res = cca_core.get_cca_similarity(
            sv_lr[:k], sv_rl[:k], epsilon=epsilon, threshold=0.98,
            compute_coefs=True, compute_dirns=False, verbose=False)
    except Exception:
        return np.zeros(n_active)

    w_lr = np.dot(res['full_coef_x'][0], res['full_invsqrt_xx']).ravel()
    w_rl = np.dot(res['coef_y'][0], res['invsqrt_yy']).ravel()
    d_lr = _cca_weights_to_neurons(acts_lr, w_lr)
    d_rl = _cca_weights_to_neurons(acts_rl, w_rl)
    d = d_lr - d_rl
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.zeros(n_active)
    return (d / norm).astype(float)


def potent_direction(X, y, active_mask, **kwargs):
    """Alias for :func:`task_potent_direction` (CCA task axis)."""
    return task_potent_direction(X, y, active_mask, **kwargs)


def _lda_project_train_test(X_train, X_test):
    """
    PCA basis from **training** crossings only; project train and test.

    When n_features < n_train - 1, returns inputs unchanged.
    """
    n_train, n_active = X_train.shape
    if n_active < n_train - 1:
        return X_train, X_test

    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_train - 2, Vt.shape[0], 50)
    k = max(1, k)
    V = Vt[:k].T
    return Xc @ V, (X_test - mu) @ V


def _lda_cv_accuracy(X, y, active_mask, splitter):
    """LDA accuracy with per-fold PCA (no test leakage into dimensionality reduction)."""
    Xs = X[:, active_mask]
    lda = LinearDiscriminantAnalysis()
    correct = total = 0
    for train_idx, test_idx in splitter.split(Xs, y):
        Z_train, Z_test = _lda_project_train_test(Xs[train_idx], Xs[test_idx])
        lda.fit(Z_train, y[train_idx])
        pred = lda.predict(Z_test)
        correct += int((pred == y[test_idx]).sum())
        total += len(test_idx)
    return correct / total


def lda_loo_accuracy(X, y, active_mask):
    """
    Leave-one-crossing-out LDA accuracy for LR vs RL.

    PCA is refit on each training set only. The test crossing is **not random**:
    ``LeaveOneOut`` uses matrix row order (LR crossings, then RL).
    """
    return _lda_cv_accuracy(X, y, active_mask, LeaveOneOut())


def lda_kfold_accuracy(X, y, active_mask, n_splits=5, random_state=0):
    """
    Stratified k-fold LDA accuracy (PCA per fold; shuffled folds).

    Returns NaN if ``n_samples < n_splits``.
    """
    if len(y) < n_splits:
        return np.nan
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    return _lda_cv_accuracy(X, y, active_mask, skf)


def _match_rows_stratified(X0, y0, Xk, yk, rng=None):
    """Subsample both matrices to the same length, balanced by class (for SVCCA)."""
    if len(y0) == len(yk):
        return X0, Xk
    rng = np.random.default_rng() if rng is None else rng
    n = min(len(y0), len(yk))
    idx0, idxk = [], []
    for label in (+1.0, -1.0):
        i0 = np.where(y0 == label)[0]
        ik = np.where(yk == label)[0]
        half = n // 2
        take = min(len(i0), len(ik), half)
        idx0.extend(rng.choice(i0, size=take, replace=len(i0) < take))
        idxk.extend(rng.choice(ik, size=take, replace=len(ik) < take))
    idx0 = np.array(idx0[:n])
    idxk = np.array(idxk[:n])
    return X0[idx0], Xk[idxk]


def potent_cosine_similarity(d0, dk):
    """
    |cos θ| between two normalised potent directions.
    Both d0 and dk must already be in the same (common-neuron) space.
    """
    if np.linalg.norm(d0) < 1e-12 or np.linalg.norm(dk) < 1e-12:
        return np.nan
    return float(np.abs(d0 @ dk))


def _principal_subspace_overlap(Xs0, Xsk, n_components=10):
    """
    Mean cos² principal angles between top-*k* subspaces (SVD, sample × feature).

    Shared by :func:`subspace_alignment` (full neuron space) and
    :func:`pca_subspace_overlap` (after SVCCA reduction).
    """
    Xs0 = Xs0 - Xs0.mean(axis=0)
    Xsk = Xsk - Xsk.mean(axis=0)

    _, _, Vt0 = scipy_svd(Xs0, full_matrices=False)
    _, _, Vtk = scipy_svd(Xsk, full_matrices=False)

    k = min(n_components, Vt0.shape[0], Vtk.shape[0])
    if k < 1:
        return np.nan

    V0 = Vt0[:k].T
    Vk = Vtk[:k].T
    svals = scipy_svd(V0.T @ Vk, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    return float(np.mean(svals ** 2))


def subspace_alignment(X0, Xk, active_mask, n_components=10):
    """
    Principal-angle alignment between top SVD subspaces of X0 and Xk
    in **neuron space** (right singular vectors of each condition matrix).

    Robust to different numbers of conditions per iteration.

    Returns
    -------
    alignment : float  [0, 1]  –  1 = identical subspaces, 0 = orthogonal
    """
    return _principal_subspace_overlap(
        X0[:, active_mask], Xk[:, active_mask], n_components=n_components)


def _first_sv_projection(Xc):
    """Project each row of ``Xc`` onto the first right singular vector (SVD)."""
    _, _, Vt = scipy_svd(Xc, full_matrices=False)
    return Xc @ Vt[0]


def within_task_direction(X, y, active_mask):
    """
    Within-iteration LR vs RL task axis: normalized mean(LR) − mean(RL).

    Used to define the task subspace when constructing a null complement for
    one crossing matrix.  Prefer this over LDA here: with few crossings and
    thousands of neurons, sklearn LDA can be nearly orthogonal to the mean
    difference and leave task structure in the putative null space.
    """
    Xs = X[:, active_mask]
    Xc = Xs - Xs.mean(axis=0)
    n_active = int(active_mask.sum())
    lr_mask = y == +1
    rl_mask = y == -1
    if lr_mask.sum() < 1 or rl_mask.sum() < 1:
        return np.zeros(n_active)
    d = Xc[lr_mask].mean(axis=0) - Xc[rl_mask].mean(axis=0)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.zeros(n_active)
    return (d / norm).astype(float)


def null_sv_direction(X, y, active_mask, task_direction=None, sv_index=0):
    """
    ``sv_index``-th right singular direction after removing ``task_direction``.

    If ``task_direction`` is omitted, uses :func:`within_task_direction`.
    """
    Xs = X[:, active_mask]
    Xc = Xs - Xs.mean(axis=0)
    d = (task_direction if task_direction is not None
         else within_task_direction(X, y, active_mask))
    Xc_null = Xc - np.outer(Xc @ d, d)
    _, _, Vt = scipy_svd(Xc_null, full_matrices=False)
    if sv_index >= Vt.shape[0]:
        return np.zeros(int(active_mask.sum()))
    v = Vt[sv_index].astype(float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros(int(active_mask.sum()))
    return v / norm


def _embed_active_direction(d, active_mask, n_total):
    """Embed an active-neuron direction into full ``N_TOTAL`` space."""
    out = np.zeros(n_total, dtype=float)
    out[active_mask] = d
    return out


def reference_null_direction(res, ref_iter=0, sv_index=0):
    """
    Null SV direction at ``ref_iter``, embedded in full neuron space.

    Task is removed with the within-iter mean(LR)−mean(RL) axis; the leading null
    SVD direction is computed on the residual crossings at ``ref_iter``.
    """
    j0 = res['iters'].index(ref_iter)
    X0, y0, mask0 = res['X'][j0], res['y'][j0], res['active_masks'][j0]
    n_total = X0.shape[1]
    d_lda = within_task_direction(X0, y0, mask0)
    v_null = null_sv_direction(X0, y0, mask0, task_direction=d_lda,
                               sv_index=sv_index)
    return _embed_active_direction(v_null, mask0, n_total)


def null_scatter_coords(X, y, active_mask, ref_null_emb, n_sv_dims=20,
                        seed=0):
    """
    2D coordinates for the task × null scatter (Fig 4 / composite panel E).

    **X:** CCA task axis (:func:`potent_direction`) — cross-set LR/RL geometry.
    **Y:** projection onto a **fixed** null direction from iter 0 (embedded in
    full neuron space, restricted to ``active_mask``).  The reference null axis
    is the leading SVD direction after removing the within-iter mean(LR)−mean(RL)
    task axis at ref iter, so it carries little LR/RL information at baseline.
    """
    Xs = X[:, active_mask]
    Xc = Xs - Xs.mean(axis=0)
    d_cca = potent_direction(X, y, active_mask, n_sv_dims=n_sv_dims,
                               seed=seed)
    v = ref_null_emb[active_mask].astype(float)
    vn = np.linalg.norm(v)
    proj_task = Xc @ d_cca
    proj_null = Xc @ (v / vn) if vn >= 1e-12 else np.zeros(len(y))
    return proj_task, proj_null


def potent_null_svd_coords(X, y, active_mask, ref_null_emb=None,
                           n_sv_dims=20, seed=0):
    """
    Backward-compatible alias for :func:`null_scatter_coords`.

    Pass ``ref_null_emb`` from :func:`reference_null_direction`; if omitted,
    falls back to the legacy per-iteration CCA-null SV1 (not recommended).
    """
    if ref_null_emb is not None:
        return null_scatter_coords(
            X, y, active_mask, ref_null_emb,
            n_sv_dims=n_sv_dims, seed=seed)
    Xs = X[:, active_mask]
    Xc = Xs - Xs.mean(axis=0)
    d = potent_direction(X, y, active_mask, n_sv_dims=n_sv_dims, seed=seed)
    proj_task = Xc @ d
    Xc_null = Xc - np.outer(proj_task, d)
    proj_null_sv1 = _first_sv_projection(Xc_null)
    return proj_task, proj_null_sv1


def svcca_similarity(X0, Xk, active_mask, threshold=0.98, match_rows=True,
                     y0=None, yk=None, seed=0):
    """
    SVCCA similarity between two condition matrices restricted to common neurons.

    Algorithm (Raghu et al. 2017):
      1. Centre each matrix.
      2. SVD on each; keep components explaining `threshold` of variance.
      3. CCA on the right-singular-vector representations.
      4. Return mean CCA correlation coefficient.

    Parameters
    ----------
    X0, Xk      : (n_cond, N_TOTAL)
    active_mask : (N_TOTAL,) boolean
    threshold   : cumulative variance threshold for SVD truncation

    Returns
    -------
    similarity : float in [0, 1], or np.nan on failure
    """
    if match_rows:
        if y0 is not None and yk is not None:
            X0, Xk = _match_rows_stratified(X0, y0, Xk, yk,
                                             rng=np.random.default_rng(seed))
        else:
            X0, Xk = _match_sample_rows(X0, Xk, seed=seed)

    Xs0 = X0[:, active_mask]
    Xsk = Xk[:, active_mask]

    # SVD in sample space when n_active >> n_samples (much faster than on X.T)
    def _truncate_svd_rows(Xc, thr):
        """Xc: (n_samples, n_active). Return (n_samples, k) condition-space PCs."""
        U, s, _ = scipy_svd(Xc, full_matrices=False)
        var = s ** 2
        var /= var.sum() + 1e-30
        k = int(np.searchsorted(np.cumsum(var), thr)) + 1
        k = max(2, min(k, len(s), Xc.shape[0] - 1))
        return U[:, :k]

    Xc0 = Xs0 - Xs0.mean(axis=0)
    Xck = Xsk - Xsk.mean(axis=0)
    Va = _truncate_svd_rows(Xc0, threshold)
    Vb = _truncate_svd_rows(Xck, threshold)

    n_comp = min(Va.shape[1], Vb.shape[1], Va.shape[0] - 1)
    #print("n_comp:", n_comp)
    if n_comp < 1:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cca = CCA(n_components=n_comp, max_iter=2000, tol=1e-6)
            cca.fit(Va, Vb)
            Vc, Vd = cca.transform(Va, Vb)
        coeffs = [abs(float(np.corrcoef(Vc[:, i], Vd[:, i])[0, 1]))
                  for i in range(n_comp)]
        return float(np.nanmean(coeffs))
    except Exception:
        return np.nan


# =============================================================================
#  2b. CCA-CORE (Raghu et al. tutorial / SVCCA notebooks)
# =============================================================================

def cca_core_pair_analysis(X0, Xk, active_mask, n_sv_dims=20,
                           epsilon=1e-10, threshold=0.98, verbose=False,
                           return_full=False):
    """
    CCA / SVCCA similarity using ``analysis/cca_core.py`` (tutorial method).

    Steps (Introduction notebook):
      1. Mean-centre activations per neuron.
      2. SVD each side; keep top ``n_sv_dims`` singular directions.
      3. Run ``get_cca_similarity`` on reduced (k × n_datapoints) matrices.

    With few corridor crossings, full neuron×datapoint CCA is ill-posed
    (more neurons than datapoints); SVCCA reduction is required.

    Returns
    -------
    dict with keys ``mean_all``, ``mean_threshold``, ``sum_coef``,
    ``n_sv_dims``, ``cca_coef1``; optionally ``results`` (full cca_core dict).
    """
    out = {
        'mean_all': np.nan,
        'mean_threshold': np.nan,
        'sum_coef': np.nan,
        'n_sv_dims': 0,
        'cca_coef1': np.array([]),
    }
    a0 = _condition_to_acts(X0, active_mask)
    ak = _condition_to_acts(Xk, active_mask)
    if a0.shape[1] != ak.shape[1] or a0.shape[1] < 3:
        return out

    k = _effective_sv_dims(a0.shape[0], a0.shape[1], n_sv_dims)
    out['n_sv_dims'] = k
    sv0, _ = _svcca_reduce_acts(a0, k)
    svk, _ = _svcca_reduce_acts(ak, k)
    k = min(sv0.shape[0], svk.shape[0])
    sv0, svk = sv0[:k], svk[:k]
    if k < 1 or sv0.shape[1] <= k:
        return out

    try:
        res = cca_core.get_cca_similarity(
            sv0, svk, epsilon=epsilon, threshold=threshold,
            compute_coefs=return_full, compute_dirns=False, verbose=verbose)
    except Exception:
        return out

    coefs = np.asarray(res['cca_coef1'], dtype=float)
    out['cca_coef1'] = coefs
    if coefs.size:
        out['mean_all'] = float(np.mean(coefs))
        out['mean_threshold'] = float(res['mean'][0])
        out['sum_coef'] = float(res['sum'][0])
    if return_full:
        out['results'] = res
        # Projections must use the same (k × n_datapoints) matrices passed to cca_core.
        out['acts0'] = sv0
        out['actsk'] = svk
    return out


def pwcca_pair_similarity(X0, Xk, active_mask, n_sv_dims=20, epsilon=1e-10):
    """Projection-weighted CCA mean (Morcos et al. 2018)."""
    a0 = _condition_to_acts(X0, active_mask)
    ak = _condition_to_acts(Xk, active_mask)
    if a0.shape[1] != ak.shape[1] or a0.shape[1] < 3:
        return np.nan
    k = _effective_sv_dims(a0.shape[0], a0.shape[1], n_sv_dims)
    sv0, _ = _svcca_reduce_acts(a0, k)
    svk, _ = _svcca_reduce_acts(ak, k)
    k = min(sv0.shape[0], svk.shape[0])
    try:
        return float(pwcca.compute_pwcca(sv0[:k], svk[:k], epsilon=epsilon)[0])
    except Exception:
        return np.nan


def pls_pair_analysis(X0, Xk, active_mask, n_sv_dims=20):
    """
    PLS similarity between two iterations (numpy_pls, notebook 003).

    Returns mean of PLS singular values on SVCCA-reduced activations.
    """
    a0 = _condition_to_acts(X0, active_mask)
    ak = _condition_to_acts(Xk, active_mask)
    if a0.shape[1] != ak.shape[1] or a0.shape[1] < 3:
        return np.nan
    k = _effective_sv_dims(a0.shape[0], a0.shape[1], n_sv_dims)
    sv0, _ = _svcca_reduce_acts(a0, k)
    svk, _ = _svcca_reduce_acts(ak, k)
    k = min(sv0.shape[0], svk.shape[0])
    try:
        res = numpy_pls.get_pls_similarity(sv0[:k], svk[:k])
        vals = np.asarray(res['eigenvals'], dtype=float)
        return float(np.mean(vals)) if vals.size else np.nan
    except Exception:
        return np.nan


def pca_subspace_overlap(X0, Xk, active_mask, n_components=10, n_sv_dims=20):
    """
    Top subspace overlap after SVCCA (SVD) reduction — same math as
    :func:`subspace_alignment`, applied in the reduced (k × crossing) space.
    """
    a0 = _condition_to_acts(X0, active_mask)
    ak = _condition_to_acts(Xk, active_mask)
    if a0.shape[1] != ak.shape[1] or a0.shape[1] < 3:
        return np.nan
    k = _effective_sv_dims(a0.shape[0], a0.shape[1], n_sv_dims)
    sv0, _ = _svcca_reduce_acts(a0, k)
    svk, _ = _svcca_reduce_acts(ak, k)
    # sv*: (k, n_crossings) → (n_crossings, k) for sample × feature SVD
    return _principal_subspace_overlap(sv0.T, svk.T, n_components=n_components)


def cca_core_projection(results_dict, side='x'):
    """
    Project SVCCA-reduced activations onto CCA directions (tutorial notebook 003).

    ``acts0`` / ``actsk`` must be the (k × n_datapoints) matrices used in
    ``get_cca_similarity`` — not full neuron-space activations.

    Returns
    -------
    proj : (n_components, n_datapoints)
    """
    res = results_dict['results']
    acts = results_dict['acts0'] if side == 'x' else results_dict['actsk']
    if side == 'x':
        coef = res['full_coef_x']
        inv = res['full_invsqrt_xx']
        means = res['neuron_means1']
    else:
        coef = res['full_coef_y']
        inv = res['full_invsqrt_yy']
        means = res['neuron_means2']
    centered = acts - means
    return np.dot(np.dot(coef.T, np.dot(coef, inv)), centered)


def _align_condition_matrices(X0, y0, Xk, yk, seed=0):
    """Subsample to equal row counts, stratified by LR / RL (for CCA pairing)."""
    if len(y0) == len(yk):
        return X0, y0, Xk, yk
    rng = np.random.default_rng(seed)
    n = min(len(y0), len(yk))
    idx0, idxk = [], []
    for label in (+1.0, -1.0):
        i0 = np.where(y0 == label)[0]
        ik = np.where(yk == label)[0]
        half = n // 2
        take = min(len(i0), len(ik), half)
        idx0.extend(rng.choice(i0, size=take, replace=len(i0) < take))
        idxk.extend(rng.choice(ik, size=take, replace=len(ik) < take))
    idx0 = np.array(idx0[:n])
    idxk = np.array(idxk[:n])
    return X0[idx0], y0[idx0], Xk[idxk], yk[idxk]


def cca_first_mode_projections(X_ref, X_k, y_ref, y_k, active_mask,
                               n_sv_dims=20, seed=0):
    """
    First canonical CCA projection for iter ``ref`` vs ``k`` (tutorial notebook 003).

    Returns
    -------
    proj_ref, proj_k : ndarray, shape (n_crossings,)
    y_ref, y_k : aligned label vectors
    ok : bool — False if CCA failed
    """
    X0, y0, Xk, yk = _align_condition_matrices(X_ref, y_ref, X_k, y_k, seed=seed)
    stats = cca_core_pair_analysis(
        X0, Xk, active_mask, n_sv_dims=n_sv_dims, return_full=True)
    if 'results' not in stats:
        return None, None, y0, yk, False
    proj = cca_core_projection(stats, side='x')
    projk = cca_core_projection(stats, side='y')
    return proj[0], projk[0], y0, yk, True


# =============================================================================
#  3. MAIN ANALYSIS LOOP
# =============================================================================

def _load_seed_lesion_data(seed_path, sc_dict, min_crossings=10, max_per_type=16):
    """Build condition matrices and kill masks for one seed."""
    active_masks = get_cumulative_killed(sc_dict)
    Xs, ys = {}, {}
    for it in sorted(sc_dict):
        if it == 0:
            subdir = 'reservoir_states'
        else:
            subdir = f'reservoir_states_1_killed_{it}'
        path = os.path.join(seed_path, subdir) + os.sep
        try:
            X, y, n_lr, n_rl = build_condition_matrix(
                path, max_per_type=max_per_type)
            if min(n_lr, n_rl) < min_crossings:
                raise ValueError(
                    f'too few crossings ({n_lr} LR, {n_rl} RL) — need ≥{min_crossings} per type')
            Xs[it] = X
            ys[it] = y
        except Exception as e:
            print(f'[skip iter {it}: {e}]', end=' ', flush=True)
    if not Xs:
        return None
    return Xs, ys, active_masks


def run_analysis(root, exclude_seeds=None, min_crossings=10, max_per_type=16):
    """Run the full subspace-alignment analysis for all seeds."""
    if exclude_seeds is None:
        exclude_seeds = {2}

    results = {}

    for seed_dir in sorted(os.listdir(root)):
        m = re.match(r'seed_(\d+)$', seed_dir)
        if not m:
            continue
        seed = int(m.group(1))
        if seed in exclude_seeds:
            continue

        print(f'  Seed {seed}…', end=' ', flush=True)
        seed_path = os.path.join(root, seed_dir)
        sc_dict   = load_splitter_indices(root, seed)
        loaded = _load_seed_lesion_data(
            seed_path, sc_dict, min_crossings, max_per_type)
        if loaded is None:
            print('skipped (no valid iterations)')
            continue
        Xs, ys, active_masks = loaded
        valid_iters = sorted(Xs)

        if 0 not in Xs:
            print('skipped (iteration 0 invalid — no baseline)')
            continue

        res = defaultdict(list)
        res['iters'] = valid_iters

        for it in valid_iters:
            mask_k = active_masks[it]
            X_k    = Xs[it]
            y_k    = ys[it]
            X_0    = Xs[0]

            # Neurons alive at iteration k (killed units are zeroed in X, not dropped).
            common_mask = mask_k

            # LDA accuracy (LOO + 10-fold + 5-fold stratified)
            res['lda_acc'].append(lda_loo_accuracy(X_k, y_k, common_mask))
            res['lda_acc_kfold10'].append(
                lda_kfold_accuracy(X_k, y_k, common_mask, n_splits=10, random_state=0))
            res['lda_acc_kfold5'].append(
                lda_kfold_accuracy(X_k, y_k, common_mask, n_splits=5, random_state=0))

            res['X'].append(X_k)
            res['y'].append(y_k)
            res['active_masks'].append(common_mask)

            if it == 0:
                res['potent_cos'].append(np.nan)
                res['subspace_align'].append(np.nan)
                res['pwcca'].append(np.nan)
                res['cca_mean'].append(np.nan)
                res['cca_mean_thr'].append(np.nan)
                res['cca_m98'].append(np.nan)
                res['pls_mean'].append(np.nan)
                res['pca_overlap'].append(np.nan)
            else:
                d_ref = task_potent_direction(X_0, ys[0], common_mask)
                dk = task_potent_direction(X_k, y_k, common_mask)
                res['potent_cos'].append(potent_cosine_similarity(d_ref, dk))

                res['subspace_align'].append(
                    subspace_alignment(X_0, X_k, common_mask, n_components=10))

                stats = cca_core_pair_analysis(
                    X_0, X_k, common_mask, n_sv_dims=20)
                res['cca_mean'].append(stats['mean_all'])
                res['cca_mean_thr'].append(stats['mean_threshold'])
                # Effective dimensionality of similarity:
                # smallest m such that sum_{i<=m} rho_i / sum_i rho_i >= 0.98
                coefs = np.asarray(stats.get('cca_coef1', np.array([])), dtype=float)
                if coefs.size and np.isfinite(coefs).any() and float(np.nansum(coefs)) > 0:
                    try:
                        m98 = cca_core.sum_threshold(coefs, 0.98)
                    except Exception:
                        m98 = None
                    if m98 is None:
                        m98 = int(coefs.size)
                    m98 = max(1, int(m98))
                else:
                    m98 = np.nan
                res['cca_m98'].append(m98)
                res['pwcca'].append(
                    pwcca_pair_similarity(X_0, X_k, common_mask))
                res['pls_mean'].append(
                    pls_pair_analysis(X_0, X_k, common_mask))
                res['pca_overlap'].append(
                    pca_subspace_overlap(X_0, X_k, common_mask))

        results[seed] = dict(res)
        print('done')

    return results


# =============================================================================
#  4. AGGREGATE HELPERS
# =============================================================================

def _aggregate(results, key, max_iter=None):
    """
    Return (all_iters, mean_per_iter, std_per_iter, n_per_iter, per_seed_dict).
    NaN values are excluded from mean/std.
    """
    # collect per-iteration across seeds
    vals_by_iter = defaultdict(list)
    for seed, res in results.items():
        for it, v in zip(res['iters'], res[key]):
            if not np.isnan(v):
                vals_by_iter[it].append(v)

    all_iters = sorted(vals_by_iter)
    if max_iter is not None:
        all_iters = [i for i in all_iters if i <= max_iter]
    means  = [np.mean(vals_by_iter[i])  for i in all_iters]
    stds   = [np.std(vals_by_iter[i])   for i in all_iters]
    ns     = [len(vals_by_iter[i])      for i in all_iters]
    # per-seed traces (NaN-padded for missing iters)
    per_seed = {}
    for seed, res in results.items():
        row = {i: v for i, v in zip(res['iters'], res[key])}
        per_seed[seed] = [row.get(i, np.nan) for i in all_iters]
    return np.array(all_iters), np.array(means), np.array(stds), np.array(ns), per_seed


def _plot_metric(ax, results, key, ylabel, color, title, show_n=True):
    """Plot mean±std + grey individual traces for one metric."""
    if not results:
        ax.text(0.5, 0.5, 'No valid seeds in results',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return

    iters, means, stds, ns, per_seed = _aggregate(results, key)
    if len(iters) == 0:
        ax.text(0.5, 0.5, f'No data for "{key}"',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return
    for seed, row in per_seed.items():
        valid = ~np.isnan(row)
        ax.plot(np.array(iters)[valid], np.array(row)[valid],
                color=GREY, lw=0.8, alpha=0.5, zorder=1)
    ax.fill_between(iters, means - stds, means + stds,
                    color=color, alpha=0.20, zorder=2)
    ax.plot(iters, means, color=color, lw=2.2, label='Mean ± std', zorder=3)
    ax.set_xlabel('Lesion iteration')
    ax.set_ylabel(ylabel)
    ax.set_xticks(iters)
    ax.set_title(title, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    if show_n:
        for it, n in zip(iters, ns):
            ax.annotate(f'N={n}', xy=(it, ax.get_ylim()[0]),
                        xytext=(0, -20), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='#555555')


def _panel_letter(ax, letter, y=1.04, fontsize=14):
    """Panel label (A, B, …) in upper-left of an axes."""
    ax.text(0.02, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='left',
            clip_on=False)


# Shared axis label in :func:`figure_reviewer_composite` (panels A, C, E).
TASK_AXIS_CCA_DIR1 = 'Task axis (CCA direction 1)'


def _populate_cca_projection_panels(axes, res, seed, ref_iter=0, n_sv_dims=20,
                                    iters_to_plot=None, ylabel='CCA direction 1'):
    """
    Draw CCA direction-1 projection scatters on a 2-D axes array (one panel per iter).

    Projections use the first shared canonical direction between ``ref_iter`` and
    each plotted iteration (``cca_first_mode_projections``).

    Used by :func:`figure4_cca_projection_scatter`.
    """
    all_iters = list(res['iters'])
    if iters_to_plot is not None:
        iters = [it for it in iters_to_plot if it in all_iters]
    else:
        iters = all_iters
    if ref_iter not in all_iters:
        return

    ref_idx = all_iters.index(ref_iter)
    X_ref = res['X'][ref_idx]
    y_ref = res['y'][ref_idx]
    later_iters = [it for it in all_iters if it > ref_iter]
    first_later = later_iters[0] if later_iters else None

    n_iters = len(iters)
    nrows, ncols = axes.shape

    for k, it in enumerate(iters):
        if k >= nrows * ncols:
            break
        ax = axes[k // ncols][k % ncols]

        if it == ref_iter:
            if first_later is None:
                ax.text(0.5, 0.5, 'No iter > ref',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Iter {it}', fontsize=8)
                continue
            j = all_iters.index(first_later)
            mask = res['active_masks'][j]
            proj_v, _, y_plot, _, ok = cca_first_mode_projections(
                X_ref, res['X'][j], y_ref, res['y'][j], mask,
                n_sv_dims=n_sv_dims, seed=seed + it)
        else:
            j = all_iters.index(it)
            mask = res['active_masks'][j]
            _, proj_v, _, y_plot, ok = cca_first_mode_projections(
                X_ref, res['X'][j], y_ref, res['y'][j], mask,
                n_sv_dims=n_sv_dims, seed=seed + it)

        if not ok or proj_v is None:
            ax.text(0.5, 0.5, 'CCA failed',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Iter {it}', fontsize=8)
            continue

        lr_mask = y_plot == +1
        rl_mask = y_plot == -1
        x_idx = np.arange(len(proj_v))
        ax.scatter(x_idx[lr_mask], proj_v[lr_mask],
                   color=BLUE, marker='o', s=40, label='LR', alpha=0.85, zorder=3)
        ax.scatter(x_idx[rl_mask], proj_v[rl_mask],
                   color=ORANGE, marker='^', s=40, label='RL', alpha=0.85, zorder=3)
        ax.set_xlabel('Crossing index', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        n_active = int(mask.sum())
        if it == ref_iter:
            ax.set_title(f'Iter {it} (ref side, vs iter {first_later})\n'
                         f'N_active={n_active}', fontsize=7)
        else:
            ax.set_title(f'Iter {it} vs iter {ref_iter}\nN_active={n_active}', fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')

    for k in range(n_iters, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)


def _populate_task_axis_projection_panels(axes, res, iters_to_plot, n_sv_dims=20,
                                        seed=0,
                                        ylabel=TASK_AXIS_CCA_DIR1):
    """
    Crossing index vs projection onto the LR/RL task axis (``potent_direction``).

    Used by :func:`figure_reviewer_composite` panel A.
    """
    all_iters = list(res['iters'])
    iters = [it for it in iters_to_plot if it in all_iters]
    n_iters = len(iters)
    nrows, ncols = axes.shape

    for k, it in enumerate(iters):
        if k >= nrows * ncols:
            break
        ax = axes[k // ncols][k % ncols]
        j = all_iters.index(it)
        X, y, mask = res['X'][j], res['y'][j], res['active_masks'][j]
        Xs = X[:, mask]
        Xc = Xs - Xs.mean(axis=0)
        d = potent_direction(X, y, mask, n_sv_dims=n_sv_dims, seed=seed + it)
        if np.linalg.norm(d) < 1e-12:
            ax.text(0.5, 0.5, 'Task axis undefined',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Iter {it}', fontsize=7)
            continue
        proj = Xc @ d
        lr_mask = y == +1
        rl_mask = y == -1
        x_idx = np.arange(len(proj))
        ax.scatter(x_idx[lr_mask], proj[lr_mask],
                   color=BLUE, marker='o', s=40, label='LR', alpha=0.85, zorder=3)
        ax.scatter(x_idx[rl_mask], proj[rl_mask],
                   color=ORANGE, marker='^', s=40, label='RL', alpha=0.85, zorder=3)
        ax.set_xlabel('Crossing index', fontsize=7)
        if k == 0:
            ax.set_ylabel(ylabel, fontsize=7)
        n_active = int(mask.sum())
        ax.set_title(f'Iter {it}\nN_active={n_active}', fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')

    for k in range(n_iters, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)


def _populate_null_scatter_panels(axes, res, iters_to_plot, ref_iter=0,
                                  n_sv_dims=20, seed=0,
                                  xlabel=TASK_AXIS_CCA_DIR1):
    """
    Task axis × reference-null scatter on a 2-D axes array (one panel per iter).

    **X:** task axis at each iteration (default LR/RL CCA direction 1).
    **Y:** projection onto iter-``ref_iter`` null SV1 (mean task removed at ref).

    Used by :func:`figure4_svd_scatter` and :func:`figure_reviewer_composite`.
    """
    all_iters = list(res['iters'])
    iters = [it for it in iters_to_plot if it in all_iters]
    n_iters = len(iters)
    nrows, ncols = axes.shape

    if ref_iter not in all_iters:
        ref_iter = all_iters[0]
    ref_null_emb = reference_null_direction(res, ref_iter=ref_iter)

    for k, it in enumerate(iters):
        if k >= nrows * ncols:
            break
        ax = axes[k // ncols][k % ncols]
        j = all_iters.index(it)
        proj_task, proj_null = null_scatter_coords(
            res['X'][j], res['y'][j], res['active_masks'][j],
            ref_null_emb, n_sv_dims=n_sv_dims, seed=seed)
        lr_mask = res['y'][j] == +1
        rl_mask = res['y'][j] == -1
        ax.scatter(proj_task[lr_mask], proj_null[lr_mask],
                   color=BLUE, marker='o', s=40, label='LR', alpha=0.85, zorder=3)
        ax.scatter(proj_task[rl_mask], proj_null[rl_mask],
                   color=ORANGE, marker='^', s=40, label='RL', alpha=0.85, zorder=3)
        ax.axvline(0, color='grey', lw=0.7, ls='--', alpha=0.6)
        ax.axhline(0, color='grey', lw=0.7, ls='--', alpha=0.6)
        ax.set_xlabel(xlabel, fontsize=7)
        if k == 0:
            ax.set_ylabel(f'Null SV1 (iter {ref_iter})', fontsize=7)
        n_active = int(res['active_masks'][j].sum())
        ax.set_title(f'Iter {it}\nN_active={n_active}', fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')

    for k in range(n_iters, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)


def _cca_spectrum_from_results(results, seed, iter_k, n_sv_dims=20,
                               ref_iter=0):
    """Return (coefs, n_sv_dims_used) for ref_iter vs iter_k, or (None, None)."""
    if seed not in results:
        return None, None
    res = results[seed]
    if ref_iter not in res['iters'] or iter_k not in res['iters']:
        return None, None
    i0 = res['iters'].index(ref_iter)
    ik = res['iters'].index(iter_k)
    stats = cca_core_pair_analysis(
        res['X'][i0], res['X'][ik], res['active_masks'][ik],
        n_sv_dims=n_sv_dims, return_full=False)
    coefs = np.asarray(stats.get('cca_coef1', []), dtype=float)
    if coefs.size == 0:
        return None, None
    return coefs, int(stats.get('n_sv_dims', n_sv_dims))


def _aggregate_cca_spectrum_all(results, ref_iter=0, n_sv_dims=20):
    """
    Pool CCA spectra (ref_iter vs each post-ref iter) over all seeds.

    Returns
    -------
    idx, means, stds : ndarray or (None, None, None) if no valid spectra
    n_spectra : int — number of (seed, iter) pairs pooled
    """
    rows = []
    for seed in results:
        for it in results[seed]['iters']:
            if it <= ref_iter:
                continue
            coefs, _ = _cca_spectrum_from_results(
                results, seed, it, n_sv_dims=n_sv_dims, ref_iter=ref_iter)
            if coefs is not None:
                rows.append(coefs)
    if not rows:
        return None, None, None, 0
    max_len = max(len(r) for r in rows)
    mat = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        mat[i, :len(r)] = r
    idx = np.arange(max_len)
    return idx, np.nanmean(mat, axis=0), np.nanstd(mat, axis=0), len(rows)


# =============================================================================
#  5. FIGURE FUNCTIONS
# =============================================================================

def figure1_lda(results, save_path=None):
    """Figure 1 — LOO, 10-fold, and 5-fold stratified LDA accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.32)

    panels = [
        ('lda_acc', BLUE,
         'A – LOO\n(train N−1, test 1)'),
        ('lda_acc_kfold10', GREEN,
         'B – 10-fold stratified\n(train ~90%, test ~3)'),
        ('lda_acc_kfold5', ORANGE,
         'C – 5-fold stratified\n(train ~80%, test ~6)'),
    ]
    for ax, (key, color, title) in zip(axes, panels):
        _plot_metric(ax, results, key, ylabel='Accuracy', color=color, title=title)
        ax.axhline(0.5, ls='--', lw=1, color='black', alpha=0.4)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Splitter-cell lesion series – LR vs RL decoding (LDA)', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure1_lda_variance(results, save_path=None):
    """Deprecated alias for :func:`figure1_lda` (variance panel removed)."""
    figure1_lda(results, save_path=save_path)


def figure2_subspace_alignment(results, save_path=None):
    """
    Figure 2 — Potent direction cosine similarity (panel A)
               + full subspace alignment / principal angles (panel B).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35)

    _plot_metric(axes[0], results, 'potent_cos',
                 ylabel='|cos θ|  (1 = same direction)',
                 color=ORANGE,
                 title='A – CCA task-axis stability\n'
                       '(cosine similarity with iteration 0)')
    axes[0].axhline(1.0, ls='--', lw=1, color=ORANGE, alpha=0.4)
    axes[0].set_ylim(0, 1.05)

    _plot_metric(axes[1], results, 'subspace_align',
                 ylabel='Alignment  (1 = identical, 0 = orthogonal)',
                 color=PURPLE,
                 title='B – Subspace alignment (SVD)\n'
                       '(principal angles, top-10 SV dirs, vs iter 0)')
    axes[1].axhline(1.0, ls='--', lw=1, color=PURPLE, alpha=0.4)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle('Splitter-cell lesion series – Subspace alignment', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure3_pwcca(results, save_path=None):
    """Figure 3 — PWCCA similarity with iteration 0."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_metric(ax, results, 'pwcca',
                 ylabel='PWCCA similarity with iter 0  [0–1]',
                 color=BLUE,
                 title='PWCCA vs iteration 0\n(cca_core + projection weights)')
    ax.axhline(1.0, ls='--', lw=1, color=BLUE, alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig.suptitle('Splitter-cell lesion series – PWCCA', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure5_cca_core(results, save_path=None):
    """
    Figure 5 — Three CCA summaries (all vs iter 0).

    Panel A: unweighted mean of all CCA correlation coefficients (after SVCCA).
    Panel B: mean only over leading coefs that carry 98% of total correlation mass
             (``cca_core`` ``sum_threshold`` / ``results['mean']``).
    Panel C: PWCCA — same coefs as A but weighted by projection onto datapoints.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _plot_metric(axes[0], results, 'cca_mean',
                 ylabel='Mean CCA coef',
                 color=BLUE,
                 title='A – Mean of all CCA coefs\n(no truncation)')
    axes[0].set_ylim(0, 1.05)

    _plot_metric(axes[1], results, 'cca_mean_thr',
                 ylabel='Mean CCA coef',
                 color=ORANGE,
                 title='B – Mean of leading CCA coefs (98% th)')
    axes[1].set_ylim(0, 1.05)

    _plot_metric(axes[2], results, 'pwcca',
                 ylabel='PWCCA',
                 color=PURPLE,
                 title='C – Projection-weighted CCA\n')
    axes[2].set_ylim(0, 1.05)

    fig.suptitle('Representation similarity vs iter 0 (cca_core / pwcca)', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure5d_cca_effective_dim(results, save_path=None):
    """
    Effective dimensionality of CCA similarity vs iter 0.

    Plots m(k): smallest number of leading canonical correlations needed to
    reach 98% of total correlation mass sum_i rho_i, for each iter 0 vs iter k.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    _plot_metric(
        ax, results, 'cca_m98',
        ylabel='m (modes for 98% cumulative ρ mass)',
        color=ORANGE,
        title='Effective shared dimensionality\n(iter 0 vs iter k, CCA 98% mass)',
        show_n=True
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure_reviewer_composite(results, seed=1, ref_iter=0, n_sv_dims=20,
                              proj_iters=(1, 3, 5),
                              null_iters=(1, 2, 3, 4, 5),
                              save_path=None):
    """
    Composite figure for reviewer response — panels A through E.

    Layout
    ------
    Row 1: **A** task axis (CCA dir. 1) vs crossing index (``proj_iters``) | **B** mean CCA coefs
    Row 2: **C** null SV1 vs task axis (CCA dir. 1) (``null_iters``, full width)
    Row 3: **D** CCA correlation spectrum | **E** task-axis rotation
    """
    if seed not in results:
        print(f'Seed {seed} not available in results.')
        return

    res = results[seed]
    if ref_iter not in res['iters']:
        print(f'Reference iter {ref_iter} not in results for seed {seed}.')
        return

    proj_iters = tuple(it for it in proj_iters if it > ref_iter)
    n_proj_a = len([it for it in proj_iters if it in res['iters']])
    null_iters = tuple(it for it in null_iters if it > ref_iter)
    n_null = len([it for it in null_iters if it in res['iters']])
    if n_proj_a == 0:
        print(f'None of proj_iters {proj_iters} available for seed {seed}.')
        return
    if n_null == 0:
        print(f'None of null_iters {null_iters} available for seed {seed}.')
        return

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.05, 1.1, 1.0], hspace=0.45)

    gs_r1 = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
    gs_a = gs_r1[0].subgridspec(1, n_proj_a, wspace=0.32)
    proj_axes = np.array([[fig.add_subplot(gs_a[0, c]) for c in range(n_proj_a)]])
    _populate_task_axis_projection_panels(
        proj_axes, res, list(proj_iters), n_sv_dims=n_sv_dims, seed=seed,
        ylabel=TASK_AXIS_CCA_DIR1)
    _panel_letter(proj_axes[0][0], 'A', y=1.10, fontsize=14)

    ax_b = fig.add_subplot(gs_r1[1])
    _plot_metric(
        ax_b, results, 'cca_mean_thr',
        ylabel='Mean CCA coef (98% threshold)',
        color=ORANGE,
        title='Mean of leading CCA coefs',
        show_n=True)
    ax_b.set_ylim(0, 1.05)
    _panel_letter(ax_b, 'B')

    gs_c = gs[1].subgridspec(1, n_null, wspace=0.28)
    null_axes = np.array([[fig.add_subplot(gs_c[0, c]) for c in range(n_null)]])
    _populate_null_scatter_panels(
        null_axes, res, list(null_iters), ref_iter=ref_iter,
        n_sv_dims=n_sv_dims, xlabel=TASK_AXIS_CCA_DIR1)
    _panel_letter(null_axes[0][0], 'C', y=1.10, fontsize=14)

    gs_r3 = gs[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
    ax_d = fig.add_subplot(gs_r3[0])
    idx, spec_mean, spec_std, n_spec = _aggregate_cca_spectrum_all(
        results, ref_iter=ref_iter, n_sv_dims=n_sv_dims)
    if idx is None:
        ax_d.text(0.5, 0.5, 'CCA spectrum unavailable',
                  ha='center', va='center', transform=ax_d.transAxes)
    else:
        ax_d.fill_between(idx, spec_mean - spec_std, spec_mean + spec_std,
                          color=BLUE, alpha=0.22, zorder=1)
        ax_d.plot(idx, spec_mean, lw=2, color=BLUE, label='Mean ± std', zorder=2)
        ax_d.set_xlabel('CCA coefficient index')
        ax_d.set_ylabel('CCA correlation')
        ax_d.set_ylim(0, 1.05)
        ax_d.grid(True, alpha=0.3)
        ax_d.set_title(f'Iter {ref_iter} vs k  (n={n_spec} seed–iter pairs)',
                       fontsize=10)
        ax_d.legend(frameon=False, fontsize=8)
    _panel_letter(ax_d, 'D')

    ax_e = fig.add_subplot(gs_r3[1])
    _plot_metric(
        ax_e, results, 'potent_cos',
        ylabel='|cos θ|',
        color=ORANGE,
        title=f'{TASK_AXIS_CCA_DIR1} rotation (vs iter 0)',
        show_n=True)
    ax_e.axhline(1.0, ls='--', lw=1, color=ORANGE, alpha=0.4)
    ax_e.set_ylim(0, 1.05)
    _panel_letter(ax_e, 'E')

    fig.suptitle(
        f'Subspace alignment summary (seed {seed})',
        fontsize=13, y=1.01)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure8_pls_pca(results, save_path=None):
    """Figure 8 — PLS mean singular value and SVD subspace overlap vs iter 0."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_metric(axes[0], results, 'pls_mean',
                 ylabel='Mean PLS singular value',
                 color=GREEN,
                 title='A – PLS (numpy_pls, magnitude-sensitive)')
    _plot_metric(axes[1], results, 'pca_overlap',
                 ylabel='SVD subspace overlap',
                 color=PURPLE,
                 title='B – Top-10 SVD subspace overlap\n(after SVCCA reduction)')
    axes[1].set_ylim(0, 1.05)
    fig.suptitle('PLS / PCA similarity vs iter 0', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure6_cca_coef_spectrum(root, seed, iter_k, exclude_seeds=None,
                              min_crossings=15, max_per_type=16, n_sv_dims=20,
                              save_path=None):
    """
    Figure 6 — CCA correlation coefficients (cca_core) for one seed/iteration.
    """
    if exclude_seeds is None:
        exclude_seeds = {2}
    if seed in exclude_seeds:
        print(f'Seed {seed} excluded.')
        return

    seed_path = os.path.join(root, f'seed_{seed}')
    sc_dict = load_splitter_indices(root, seed)
    loaded = _load_seed_lesion_data(
        seed_path, sc_dict, min_crossings, max_per_type)
    if loaded is None or 0 not in loaded[0]:
        print('No valid data for spectrum plot.')
        return
    Xs, ys, active_masks = loaded
    if iter_k not in Xs:
        print(f'Iteration {iter_k} not available.')
        return

    X_ref, it_ref = Xs[0], 0
    mask = active_masks[iter_k]

    stats = cca_core_pair_analysis(
        X_ref, Xs[iter_k], mask, n_sv_dims=n_sv_dims, return_full=False)
    coefs = stats['cca_coef1']
    if coefs.size == 0:
        print('CCA failed for this pair.')
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coefs, lw=2, color=BLUE)
    ax.set_xlabel('CCA coefficient index')
    ax.set_ylabel('CCA correlation')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Seed {seed}: iter {it_ref} vs {iter_k}  '
                 f'(k={stats["n_sv_dims"]} SV dims)')
    fig.suptitle('CCA coefficient spectrum (cca_core)', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure7_cca_projection(root, seed, iter_k, exclude_seeds=None,
                           min_crossings=15, max_per_type=16, n_sv_dims=20,
                           save_path=None):
    """
    Figure 7 — First CCA direction projected onto datapoints (tutorial Fig 003).
    """
    if exclude_seeds is None:
        exclude_seeds = {2}

    seed_path = os.path.join(root, f'seed_{seed}')
    sc_dict = load_splitter_indices(root, seed)
    loaded = _load_seed_lesion_data(
        seed_path, sc_dict, min_crossings, max_per_type)
    if loaded is None:
        return
    Xs, ys, active_masks = loaded
    if 0 not in Xs or iter_k not in Xs:
        return
    X_ref, it_ref = Xs[0], 0
    mask = active_masks[iter_k]
    proj0, projk, y0, yk, ok = cca_first_mode_projections(
        X_ref, Xs[iter_k], ys[it_ref], ys[iter_k], mask, n_sv_dims=n_sv_dims,
        seed=seed)
    if not ok:
        print('CCA projection unavailable (computation failed).')
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, v, y, title in zip(
            axes,
            (proj0, projk),
            (y0, yk),
            (f'Iter {it_ref} (reference)', f'Iter {iter_k}')):
        lr, rl = y == 1, y == -1
        x_idx = np.arange(len(v))
        ax.scatter(x_idx[lr], v[lr], c=BLUE, marker='o', s=40,
                   label='LR', alpha=0.85, zorder=3)
        ax.scatter(x_idx[rl], v[rl], c=ORANGE, marker='^', s=40,
                   label='RL', alpha=0.85, zorder=3)
        ax.set_xlabel('Crossing index')
        ax.set_ylabel('CCA direction 1')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'CCA projections (cca_core) – seed {seed}', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure8_pls_projection(root, seed, iter_k, exclude_seeds=None,
                           min_crossings=15, max_per_type=16, n_sv_dims=20,
                           save_path=None):
    """Figure 8b — First PLS direction on crossings (numpy_pls, notebook 003)."""
    if exclude_seeds is None:
        exclude_seeds = {2}
    seed_path = os.path.join(root, f'seed_{seed}')
    sc_dict = load_splitter_indices(root, seed)
    loaded = _load_seed_lesion_data(
        seed_path, sc_dict, min_crossings, max_per_type)
    if loaded is None or 0 not in loaded[0] or iter_k not in loaded[0]:
        return
    Xs, ys, active_masks = loaded
    mask = active_masks[iter_k]
    a0 = _condition_to_acts(Xs[0], mask)
    ak = _condition_to_acts(Xs[iter_k], mask)
    k = _effective_sv_dims(a0.shape[0], a0.shape[1], n_sv_dims)
    sv0, _ = _svcca_reduce_acts(a0, k)
    svk, _ = _svcca_reduce_acts(ak, k)
    k = min(sv0.shape[0], svk.shape[0])
    try:
        res = numpy_pls.get_pls_similarity(sv0[:k], svk[:k])
    except Exception:
        print('PLS projection failed.')
        return
    proj0 = np.dot(
        res['neuron_coeffs1'].T,
        np.dot(res['neuron_coeffs1'], sv0[:k] - sv0[:k].mean(axis=1, keepdims=True)))
    projk = np.dot(
        res['neuron_coeffs2'].T,
        np.dot(res['neuron_coeffs2'], svk[:k] - svk[:k].mean(axis=1, keepdims=True)))
    y = ys[iter_k]
    lr, rl = y == 1, y == -1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, proj, title in zip(
            axes, (proj0, projk), (f'Iter 0', f'Iter {iter_k}')):
        v = proj[0]
        x_idx = np.arange(len(v))
        ax.scatter(x_idx[lr], v[lr], c=BLUE, marker='o', s=40, label='LR', zorder=3)
        ax.scatter(x_idx[rl], v[rl], c=ORANGE, marker='^', s=40, label='RL', zorder=3)
        ax.set_xlabel('Crossing index')
        ax.set_ylabel('PLS direction 1')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'PLS projections – seed {seed}', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure4_svd_scatter(results, seed=1, ref_iter=0, n_sv_dims=20,
                      save_path=None):
    """
    Figure 4 — scatter of corridor crossings in task + null SVD coordinates.

    **X:** CCA task axis (``potent_direction``).
    **Y:** projection onto iter-``ref_iter`` null SV1 (mean task removed at ref).

    LR → blue circles; RL → orange triangles.
    """
    if seed not in results:
        print(f'Seed {seed} not available in results.')
        return

    res = results[seed]
    n_iters = len(res['iters'])
    ncols = min(n_iters, 6)
    nrows = int(np.ceil(n_iters / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    _populate_null_scatter_panels(
        axes, res, list(res['iters']), ref_iter=ref_iter,
        n_sv_dims=n_sv_dims)

    fig.suptitle(
        f'Task × ref-null scatter (null SV1 from iter {ref_iter}), seed {seed}',
        fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure4_pca_scatter(results, seed=1, save_path=None):
    """Deprecated alias for :func:`figure4_svd_scatter`."""
    figure4_svd_scatter(results, seed=seed, save_path=save_path)


def figure4_cca_projection_scatter(results, seed=1, ref_iter=0, n_sv_dims=20,
                                   save_path=None):
    """
    Fig 4-style grid: CCA mode-1 projection per lesion iteration (Fig 7 content).

    For each iteration *k*, CCA is run between ``ref_iter`` (default 0) and *k*
    on SVCCA-reduced crossing activations. Each panel scatters crossings vs
    projection on the **shared first canonical direction**:

    - **Iter 0:** projection of baseline crossings (ref side of CCA vs iter 1).
    - **Iter k ≥ 1:** projection of iter-*k* crossings (k side), same CCA basis.

    Layout matches :func:`figure4_svd_scatter` (one panel per iteration).
    """
    if seed not in results:
        print(f'Seed {seed} not available in results.')
        return

    res = results[seed]
    iters = list(res['iters'])
    if ref_iter not in iters:
        print(f'Reference iter {ref_iter} not in results for seed {seed}.')
        return

    n_iters = len(iters)
    ncols = min(n_iters, 6)
    nrows = int(np.ceil(n_iters / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    _populate_cca_projection_panels(
        axes, res, seed, ref_iter=ref_iter, n_sv_dims=n_sv_dims)

    fig.suptitle(
        f'CCA mode-1 projections (ref iter {ref_iter}), seed {seed}',
        fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def figure4_cca_projection_scatter_from_root(
        root, seed, exclude_seeds=None, min_crossings=15, max_per_type=16,
        ref_iter=0, n_sv_dims=20, save_path=None):
    """Run analysis for one seed and plot :func:`figure4_cca_projection_scatter`."""
    if exclude_seeds is None:
        exclude_seeds = {2}
    if seed in exclude_seeds:
        print(f'Seed {seed} excluded.')
        return
    seed_path = os.path.join(root, f'seed_{seed}')
    sc_dict = load_splitter_indices(root, seed)
    loaded = _load_seed_lesion_data(
        seed_path, sc_dict, min_crossings, max_per_type)
    if loaded is None or ref_iter not in loaded[0]:
        print(f'No valid data for seed {seed}.')
        return
    Xs, ys, masks = loaded
    res = {
        'iters': sorted(Xs),
        'X': [Xs[it] for it in sorted(Xs)],
        'y': [ys[it] for it in sorted(Xs)],
        'active_masks': [masks[it] for it in sorted(Xs)],
    }
    figure4_cca_projection_scatter(
        {seed: res}, seed=seed, ref_iter=ref_iter, n_sv_dims=n_sv_dims,
        save_path=save_path)


# =============================================================================
#  6. MAIN
# =============================================================================

if __name__ == '__main__':
    ROOT = '../data/R-L/no_cues/3000_units_lr_04'
    SAVE_DIR = '../data/R-L/no_cues/3000_units_lr_04'   # set to None to not save
    EXCLUDE  = {}

    print('Running subspace alignment analysis …')
    results = run_analysis(
        ROOT, exclude_seeds=EXCLUDE, min_crossings=15, max_per_type=16)

    #print('\nPlotting Figure 1 (LDA) …')
    #save = os.path.join(SAVE_DIR, 'fig1_lda.pdf') if SAVE_DIR else None
    #figure1_lda(results, save_path=save)

    print('Plotting Figure 2 (subspace alignment) …')
    save = os.path.join(SAVE_DIR, 'fig2_subspace_alignment.pdf') if SAVE_DIR else None
    #figure2_subspace_alignment(results, save_path=save)

    #print('Plotting Figure 3 (PWCCA) …')
    #save = os.path.join(SAVE_DIR, 'fig3_pwcca.pdf') if SAVE_DIR else None
    #figure3_pwcca(results, save_path=save)


    #print('Plotting Figure 4 (PCA scatter, seed 1) …')
    #for k in range(1, 11):
    #    save = os.path.join(SAVE_DIR, f'fig4_pca_scatter_seed{k}.pdf') if SAVE_DIR else None
    #    figure4_svd_scatter(results, seed=k, save_path=save)
        

    print('Plotting Figure 5 (CCA summaries) …')
    save = os.path.join(SAVE_DIR, 'fig5_cca_core.pdf') if SAVE_DIR else None
    #figure5_cca_core(results, save_path=save)

    print('Plotting Figure 5d (CCA effective dimensionality m98) …')
    save = os.path.join(SAVE_DIR, 'fig5d_cca_m98.pdf') if SAVE_DIR else None
    #figure5d_cca_effective_dim(results, save_path=save)

    print('Plotting Figure 8 (PLS / PCA) …')
    save = os.path.join(SAVE_DIR, 'fig8_pls_pca.pdf') if SAVE_DIR else None
    #figure8_pls_pca(results, save_path=save)

    print('Plotting Figure 6 (CCA coef spectrum, seed 1 iter 5) …')
    save = os.path.join(SAVE_DIR, 'fig6_cca_spectrum_seed3.pdf') if SAVE_DIR else None
    #figure6_cca_coef_spectrum(
    #    ROOT, seed=3, iter_k=5, exclude_seeds=EXCLUDE, save_path=save)

    print('Plotting Figure 7 (CCA projection, seed 1 iter 5) …')
    save = os.path.join(SAVE_DIR, 'fig7_cca_projection_seed3.pdf') if SAVE_DIR else None
    #figure7_cca_projection(
    #    ROOT, seed=3, iter_k=5, exclude_seeds=EXCLUDE, save_path=save)

    print('Plotting Figure 4b (CCA projection grid, seed 1) …')
    save = os.path.join(SAVE_DIR, 'fig4_cca_projection_seed1.pdf') if SAVE_DIR else None
    #figure4_cca_projection_scatter(results, seed=1, save_path=save)

    print('Plotting reviewer composite figure (panels A–E) …')
    save = os.path.join(SAVE_DIR, 'fig_reviewer_composite_seed1.pdf') if SAVE_DIR else None
    figure_reviewer_composite(
        results, seed=1, proj_iters=(1, 3, 5),
        null_iters=(1, 2, 3, 4, 5), save_path=save)

