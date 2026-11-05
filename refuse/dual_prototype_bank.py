"""
Two class-prototype banks for ReFuSe Stage 2.

  1. Target prototypes  (19, C): target-model features (DeepLabV2 layer4, C=2048; MiT-B5 fuse_layer, C=256),
     averaged over reliable (Case 1) SAM2 regions.
  2. VLM prototypes (19, 512): VLM (CorrCLIP / ProxyCLIP) CLIP-space features, averaged over
     the same Case 1 regions.

Prototypes measure how well a region's features fit a candidate class
label. They never produce new predictions -- they only arbitrate between
the two existing candidates (y_s from the target model, y_c from the VLM)
for a disagreement region.

Everything here is stored and returned detached; the bank never enters the
autograd graph.

Deviations from the original spec (all deliberate, see comments inline):
  - Separate per-bank counts (target_counts / clip_counts). With one shared
    proto_counts incremented by both update_target and update_clip, the
    second update of a class's first region would see count > 0 and EMA
    the CLIP prototype toward it from the zero init instead of copying it.
    `proto_counts` is kept as a read-only view: the number of regions
    that updated BOTH banks for a class (min of the two counts).
  - sigma is the mean SQUARED distance of prototypes to their mean, not
    the mean distance. compute_compatibility divides a squared distance by
    sigma, so sigma must also be squared for exp(-dist / sigma) to be
    scale-free.

All features are L2-normalised: on update (so each prototype is an EMA of
unit vectors) and again, together with the prototype, before every
distance. Distances and sigma therefore live on the unit sphere, and the
raw magnitude of target features (norms of ~10-30) plays no role.
"""

import torch
import torch.nn.functional as F

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain',
               'sky', 'person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle']

MIN_READY_CLASSES = 10
EPS = 1e-8


class DualPrototypeBank:
    def __init__(self, num_classes=19, target_feat_dim=2048,
                 clip_feat_dim=512, ema_momentum=0.99, device='cuda'):
        self.num_classes = num_classes
        self.target_feat_dim = target_feat_dim
        self.clip_feat_dim = clip_feat_dim
        self.ema_momentum = ema_momentum
        self.device = torch.device(device)

        self.target_protos = torch.zeros(num_classes, target_feat_dim, device=self.device)
        self.clip_protos = torch.zeros(num_classes, clip_feat_dim, device=self.device)
        self.target_counts = torch.zeros(num_classes, dtype=torch.long, device=self.device)
        self.clip_counts = torch.zeros(num_classes, dtype=torch.long, device=self.device)
        self.target_sigma = 0.0
        self.clip_sigma = 0.0
        self.initialized = False

    @property
    def proto_counts(self):
        """Regions that updated both banks, per class."""
        return torch.minimum(self.target_counts, self.clip_counts)

    # ── Updates ──────────────────────────────────────────────────────────
    @staticmethod
    def _ema_update(protos, counts, feat_region, class_label, momentum):
        feat_region = F.normalize(feat_region.detach().to(protos.device, protos.dtype), dim=0)
        if counts[class_label] == 0:
            protos[class_label] = feat_region
        else:
            protos[class_label] = momentum * protos[class_label] + (1 - momentum) * feat_region
        counts[class_label] += 1

    @staticmethod
    def _spread(protos, counts):
        """Mean squared distance of the initialized (normalised) prototypes
        to their mean -- same space compute_compatibility measures in."""
        active = F.normalize(protos[counts > 0], dim=1)
        if active.shape[0] < 2:
            return 0.0
        return ((active - active.mean(dim=0, keepdim=True)) ** 2).sum(dim=1).mean().item()

    @torch.no_grad()
    def update_target(self, feat_region, class_label):
        """feat_region: (target_feat_dim,) mean target feature of one Case 1 region."""
        class_label = int(class_label)
        self._ema_update(self.target_protos, self.target_counts, feat_region,
                         class_label, self.ema_momentum)
        self.target_sigma = self._spread(self.target_protos, self.target_counts)
        self.initialized = self.is_ready()

    @torch.no_grad()
    def update_clip(self, feat_region, class_label):
        """feat_region: (clip_feat_dim,) mean VLM feature of one Case 1 region."""
        class_label = int(class_label)
        self._ema_update(self.clip_protos, self.clip_counts, feat_region,
                         class_label, self.ema_momentum)
        self.clip_sigma = self._spread(self.clip_protos, self.clip_counts)
        self.initialized = self.is_ready()

    def is_ready(self):
        """True once at least MIN_READY_CLASSES classes have both prototypes."""
        return int((self.proto_counts > 0).sum()) >= MIN_READY_CLASSES

    # ── Scoring ──────────────────────────────────────────────────────────
    @staticmethod
    def _valid_distances(feat_norm, protos, counts):
        """(valid class ids, squared L2 distances) from the L2-normalised query
        feature to every initialised (count > 0) L2-normalised prototype.
        Both sides are unit vectors, so every distance lies in [0, 4]."""
        valid = torch.nonzero(counts > 0, as_tuple=True)[0]
        dists = ((F.normalize(protos[valid], dim=1) - feat_norm[None]) ** 2).sum(dim=1)
        return valid, dists

    @torch.no_grad()
    def _rank_score(self, feat_norm, class_id, protos, counts):
        """Rank of class_id among all initialised prototypes of one bank, by
        distance to feat_norm, as a score in [0, 1]: (N - rank) / (N - 1),
        rank 1 = closest -> 1.0, rank N = furthest -> 0.0. Only the ordering
        matters, so the score is on the same scale for the 2048-d target bank
        and the 512-d CLIP bank. 0.5 (neutral) if class_id has no prototype
        or fewer than 2 classes have one."""
        class_id = int(class_id)
        if counts[class_id] == 0:
            return 0.5
        valid, dists = self._valid_distances(feat_norm, protos, counts)
        n = valid.numel()
        if n < 2:
            return 0.5
        order = valid[torch.argsort(dists)]                        # ascending distance
        rank = int(torch.nonzero(order == class_id, as_tuple=True)[0]) + 1
        return (n - rank) / (n - 1)

    @torch.no_grad()
    def _best_class_in_bank(self, feat_norm, protos, counts):
        """Class id of the closest initialised prototype, None if there is
        none. Interface for third-class detection (not used yet)."""
        valid, dists = self._valid_distances(feat_norm, protos, counts)
        if valid.numel() == 0:
            return None
        return int(valid[torch.argmin(dists)])

    @torch.no_grad()
    def compute_compatibility(self, f_target_region, f_clip_region, y_s, y_c):
        """Rank-based, cross-model evidence for the two candidate labels.
        Returns (score_ys, score_yc) in [0, 1] -- the training scripts still
        call them s_s, s_c:

            rank_s_own   = rank of y_s in the TARGET bank, target feature
            rank_s_cross = rank of y_s in the CLIP bank,   CLIP feature
            rank_c_own   = rank of y_c in the CLIP bank,   CLIP feature
            rank_c_cross = rank of y_c in the TARGET bank, target feature
            score_ys = (rank_s_own + rank_s_cross) / 2
            score_yc = (rank_c_own + rank_c_cross) / 2

        Both candidates are judged by both models, on the same rank scale,
        regardless of feature dimension. A rank whose class has no prototype
        in that bank is the neutral 0.5 (from _rank_score). (None, None) only
        when neither y_s nor y_c has a prototype in either bank."""
        r = self.rank_terms(f_target_region, f_clip_region, y_s, y_c)
        if r is None:
            return None, None
        score_ys = (r['rank_s_own'] + r['rank_s_cross']) / 2
        score_yc = (r['rank_c_own'] + r['rank_c_cross']) / 2
        return score_ys, score_yc

    @torch.no_grad()
    def rank_terms(self, f_target_region, f_clip_region, y_s, y_c):
        """The four rank scores behind compute_compatibility, as a dict with
        keys rank_s_own, rank_s_cross, rank_c_own, rank_c_cross (see there).
        Exposed so a caller can check the cross-model terms on their own.
        None when neither y_s nor y_c has a prototype in either bank."""
        y_s, y_c = int(y_s), int(y_c)
        has_proto = lambda y: bool(self.target_counts[y] > 0 or self.clip_counts[y] > 0)
        if not (has_proto(y_s) or has_proto(y_c)):
            return None

        f_t = F.normalize(f_target_region.detach().to(self.device, self.target_protos.dtype), dim=0)
        f_c = F.normalize(f_clip_region.detach().to(self.device, self.clip_protos.dtype), dim=0)
        return {
            'rank_s_own':   self._rank_score(f_t, y_s, self.target_protos, self.target_counts),
            'rank_s_cross': self._rank_score(f_c, y_s, self.clip_protos, self.clip_counts),
            'rank_c_own':   self._rank_score(f_c, y_c, self.clip_protos, self.clip_counts),
            'rank_c_cross': self._rank_score(f_t, y_c, self.target_protos, self.target_counts),
        }

    @torch.no_grad()
    def class_posteriors(self, f_target_region, f_clip_region, temperature):
        """Discriminative alternative to compute_compatibility (used by the
        EMA-teacher FINAL stage): (p_t, p_c), each a (num_classes,) softmax of
        cosine similarity / temperature over every initialised prototype of
        that bank (uninitialised classes get probability 0). Unlike the
        single-prototype distance, this says how well a region fits a class
        RELATIVE to the other classes, and both banks yield probabilities on
        the same scale, so they can be compared. None while the bank isn't
        ready."""
        if not self.is_ready():
            return None
        f_t = F.normalize(f_target_region.detach().to(self.device, self.target_protos.dtype), dim=0)
        f_c = F.normalize(f_clip_region.detach().to(self.device, self.clip_protos.dtype), dim=0)
        cos_t = F.normalize(self.target_protos, dim=1) @ f_t
        cos_c = F.normalize(self.clip_protos, dim=1) @ f_c
        cos_t = cos_t.masked_fill(self.target_counts == 0, float('-inf'))
        cos_c = cos_c.masked_fill(self.clip_counts == 0, float('-inf'))
        return F.softmax(cos_t / temperature, dim=0), F.softmax(cos_c / temperature, dim=0)

    @staticmethod
    def compute_delta(s_s, s_c):
        """Feature discrimination score in [0, 1]: 0 = both candidates fit
        equally well, 1 = only one fits."""
        return abs(s_s - s_c) / (s_s + s_c + EPS)

    def get_target_proto(self, class_label):
        """Unit-norm target prototype (an EMA of unit vectors is slightly
        shorter than 1, so it is re-normalised)."""
        return F.normalize(self.target_protos[int(class_label)].detach(), dim=0)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path):
        torch.save(self.state_dict(), path)

    def state_dict(self):
        """Also usable to embed the bank inside a training checkpoint dict."""
        return {
            'target_protos': self.target_protos,
            'clip_protos':   self.clip_protos,
            'target_counts': self.target_counts,
            'clip_counts':   self.clip_counts,
            'target_sigma':  self.target_sigma,
            'clip_sigma':    self.clip_sigma,
            'ema_momentum':  self.ema_momentum,
        }

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def load_state_dict(self, state):
        self.target_protos = state['target_protos'].to(self.device)
        self.clip_protos = state['clip_protos'].to(self.device)
        self.target_counts = state['target_counts'].to(self.device)
        self.clip_counts = state['clip_counts'].to(self.device)
        self.target_sigma = float(state['target_sigma'])
        self.clip_sigma = float(state['clip_sigma'])
        self.ema_momentum = state.get('ema_momentum', self.ema_momentum)
        self.num_classes, self.target_feat_dim = self.target_protos.shape
        self.clip_feat_dim = self.clip_protos.shape[1]
        self.initialized = self.is_ready()

    def summary(self):
        print(f'{"class":<14}{"t_count":>9}{"c_count":>9}{"target_norm":>13}{"clip_norm":>11}')
        t_norms = self.target_protos.norm(dim=1).tolist()
        c_norms = self.clip_protos.norm(dim=1).tolist()
        for c in range(self.num_classes):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            print(f'{name:<14}{int(self.target_counts[c]):>9}{int(self.clip_counts[c]):>9}'
                  f'{t_norms[c]:>13.4f}{c_norms[c]:>11.4f}')
        print(f'initialized={self.is_ready()}  '
              f'(classes with both prototypes: {int((self.proto_counts > 0).sum())}/{self.num_classes}, '
              f'need {MIN_READY_CLASSES})')
        print(f'target_sigma={self.target_sigma:.6f}  clip_sigma={self.clip_sigma:.6f}')



if __name__ == '__main__':
    torch.manual_seed(0)
    bank = DualPrototypeBank(num_classes=19, device='cpu')
    for c in range(10):
        bank.target_protos[c] = F.normalize(torch.randn(2048), dim=0)
        bank.clip_protos[c] = F.normalize(torch.randn(512), dim=0)
    bank.target_counts[:10] = 1
    bank.clip_counts[:10] = 1

    def near(proto, away=(), noise=0.3):
        """Unit vector close to proto (plus noise), optionally pushed away
        from the prototypes in `away`."""
        v = proto + noise * F.normalize(torch.randn_like(proto), dim=0)
        for a in away:
            v = v - 0.2 * a
        return F.normalize(v, dim=0)

    # 1) Region is road (0): y_s = road (right), y_c = sidewalk (wrong).
    ft = near(bank.target_protos[0])
    fc = near(bank.clip_protos[0])
    s_ys, s_yc = bank.compute_compatibility(ft, fc, y_s=0, y_c=1)
    print(f'road region,  y_s=road, y_c=sidewalk: score_ys={s_ys:.3f}  score_yc={s_yc:.3f}')
    assert s_ys > s_yc

    # 2) Region is pole (5): both candidates wrong. With random prototypes the
    # ranks of road/sidewalk among the 9 non-pole classes are arbitrary, so the
    # region is also moved slightly away from them -- a pole region is not
    # expected to look like road or sidewalk.
    ft = near(bank.target_protos[5], away=(bank.target_protos[0], bank.target_protos[1]))
    fc = near(bank.clip_protos[5], away=(bank.clip_protos[0], bank.clip_protos[1]))
    s_ys, s_yc = bank.compute_compatibility(ft, fc, y_s=0, y_c=1)
    print(f'pole region,  y_s=road, y_c=sidewalk: score_ys={s_ys:.3f}  score_yc={s_yc:.3f}')
    assert s_ys < 0.4 and s_yc < 0.4

    # Helpers: best class, neutral score for a class without prototype, None case.
    assert bank._best_class_in_bank(fc, bank.clip_protos, bank.clip_counts) == 5
    assert bank._best_class_in_bank(fc, bank.clip_protos, torch.zeros(19, dtype=torch.long)) is None
    assert bank._rank_score(ft, 12, bank.target_protos, bank.target_counts) == 0.5
    assert bank.compute_compatibility(ft, fc, y_s=12, y_c=15) == (None, None)
    print('PASS')
