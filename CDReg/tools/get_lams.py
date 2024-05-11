"""
Calculate optimal lambda list for hyper-parameter searching
lam1 = lam * alpha
lam2 = lam * (1 - alpha)
"""

import torch


def grad_at_zero(model, X, Y, pair, lam_con):
    # Compute beta[idx_g] gradient, when beta = 0
    n = len(Y)
    Y = Y.float().view(-1, 1)

    # Step 1. Get loss_obj grad, with beta = 0
    beta_grad_obj = - 1 / n * X.t().mm(Y)

    # Step 2. Get contrastive gradient, with beta = 0
    if lam_con == 0:
        return beta_grad_obj
    else:
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        beta_true = model.beta.data.clone()
        model.beta.data *= 0.
        loss_con = lam_con * model.get_Lcon_neg_mask(X, pair)
        loss_con.backward()
        beta_grad_con = model.beta.grad.clone()
        model.beta.grad.zero_()
        model.beta.data = beta_true

        return beta_grad_obj + beta_grad_con


def get_Lams(G, gp_idx_list, alpha=0.95, min_frac=0.05, nlam=10):
    # Get regularization path: Lams_1, Lams_2 with log-linear steps
    assert 0 <= alpha < 1, 'alpha must be in [0,1).'
    assert min_frac < 1, 'min_frac must be in (0,1).'
    if alpha == 0.5:
        alpha = 0.5 + 1e-6
    Lams = torch.zeros(len(gp_idx_list))

    for num_g, idx_g in enumerate(gp_idx_list):
        G_tmp = G[idx_g]
        G_ord = G_tmp.abs().sort(descending=True).values

        if alpha == 0:
            Lams[num_g] = G_ord.norm(p=2) / (len(idx_g) ** 0.5)

        elif len(G_ord) == 1:
            # In this case, len(idx_g) = 1, alpha + (1 - alpha) * sqrt(1) = 1, hence L12 & L1 merges.
            Lams[num_g] = G_ord

        elif len(G_ord) > 1:
            G_norms = torch.zeros(len(G_ord) - 1).to(G_ord.device)
            lam_candidates = G_ord / alpha

            for j in range(1, len(G_ord)):
                # threshold sliding
                G_norms[j - 1] = (G_ord[:j] - G_ord[j]).norm(p=2)

            # Case 1. Largest threshold is NOT sufficient
            if G_norms[0] >= (lam_candidates[1] * (1 - alpha) * (len(idx_g) ** 0.5)):
                G_active = G_ord[0]
                lam_range = [G_ord[1] / alpha, G_ord[0] / alpha]  # [lb, ub]

            # Case 2. Smallest threshold is sufficient
            elif G_norms[-1] <= (lam_candidates[-1] * (1 - alpha) * (len(idx_g) ** 0.5)):
                G_active = G_ord
                lam_range = [0., G_ord[-1] / alpha]

            # Case 3. Optimal threshold is at the middle
            else:
                L = len(G_norms)
                opt_ind = torch.where(G_norms[:-1] <= (lam_candidates[1:L] * (1 - alpha) * (len(idx_g) ** 0.5)))[0]
                opt_ind = opt_ind.max() + 1
                G_active = G_ord[:opt_ind]
                lam_range = [G_ord[opt_ind + 1] / alpha, G_ord[opt_ind] / alpha]

            num_active = len(G_active)

            A_term = num_active * alpha ** 2 - (1 - alpha) ** 2 * len(idx_g)
            B_term = - 2 * alpha * G_active.sum()
            C_term = (G_active ** 2).sum()

            lam_candidates = torch.zeros(2).to(G_ord.device)
            lam_candidates[0] = (- B_term + (B_term ** 2 - 4 * A_term * C_term).sqrt()) / (2 * A_term)
            lam_candidates[1] = (- B_term - (B_term ** 2 - 4 * A_term * C_term).sqrt()) / (2 * A_term)

            lam_candidates = lam_candidates[(lam_range[0] <= lam_candidates) * (lam_candidates <= lam_range[1])]
            assert len(lam_candidates) > 0, 'lam out of range, my program is wrong.'
            Lams[num_g] = lam_candidates.min()

    Lam_max = Lams.max()
    Lam_min = min_frac * Lam_max

    # Log-linear step size
    Lams = torch.linspace(start=Lam_min.log().item(),
                          end=Lam_max.log().item(),
                          steps=nlam).exp()

    Lams_1 = alpha * Lams
    Lams_2 = (1 - alpha) * Lams

    return Lams_1, Lams_2
