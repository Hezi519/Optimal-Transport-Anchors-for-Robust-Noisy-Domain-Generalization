# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from asyncio.log import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import ot

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks, hparams_registry
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj
)



ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'FocalERM',
    'SIRM',
    'ERMReg',
    'ERM',
    'OT', 
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        if self.hparams.get('feature_noise') is not None and self.hparams['feature_noise'] is not None:
            print("Inject feature noise!")
            self.feature_noise = FeatureNoise(self.hparams['feature_noise'])
            self.network = nn.Sequential(
                self.featurizer,
                self.feature_noise,
                self.classifier
            )
        else:
            self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Add gradient noise if specified in hyperparameters
        grad_noise = self.hparams.get('grad_noise', 0)
        if grad_noise is not None and grad_noise > 0:
            for param in self.network.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * grad_noise
                    param.grad.add_(noise)
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class NLPGERM(ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(NLPGERM, self).__init__(input_shape, num_classes, num_domains,
                                      hparams)
        self.num_classes = num_classes

        # Initialize Moving Average (MA) network
        self.network_ma = copy.deepcopy(self.network)
        self.network_ma.eval()
        self.ma_start_iter = 100
        self.global_iter = 0
        self.ma_count = 0

    def set_nlp_anchor(self, dataset, data_name = 'default'):
        """
        Sets NLP anchors using CLIP and averages multiple text prompts per class.
        """
        # # for wildbirds
        # if data_name != 'default':
        #     classes_names = dataset.classes
        # if "fmow" in dataset.dataset_name:
        #     # print(dir(dataset))  # ['CHECKPOINT_FREQ', 'ENVIRONMENTS', 'INPUT_SHAPE', 'N_STEPS', 'N_WORKERS', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'data_noise', 'dataset_name', 'datasets', 'flip_prob', 'grouped_test_datasets', 'grouped_val_datasets', 'hard_datasets', 'input_shape', 'metadata_values', 'noisy_dataset_ids', 'noisy_datasets', 'num_classes', 'test_group_indices', 'torch_bernoulli_', 'torch_xor_']
        #     classes_names = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]

        #     # classes_names = dataset 
        # else:
        classes_names = dataset.datasets[0].classes
        
        import clip
        device = next(self.network.parameters()).device
        model, preprocess = clip.load("RN50", device)

        # Generate multiple prompts per class
        text_prompts = [
            [
                f"a photo of a {item}"
            ]
            for item in classes_names
        ]
        print(text_prompts)

        all_text_features = []

        for prompts in text_prompts:
            # Tokenize all prompts for the class
            text_tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                # Encode each prompt and average the embeddings
                text_features = model.encode_text(text_tokens)
                text_features_avg = text_features.mean(dim=0)  # Average the features
            all_text_features.append(text_features_avg)

        # Stack the averaged features to form the anchor matrix
        self.nlpanchor = torch.stack(all_text_features)

        # Define mapping layers
        self.maplayers = nn.ModuleList([
            nn.Linear(self.featurizer.n_outputs, len(self.nlpanchor[0])).to(device)
            for _ in range(self.num_classes)
        ])

        # Optimizer setup
        if self.hparams['mapsty'] == 'all':
            self.optimizer = torch.optim.Adam(
                list(self.network.parameters()) + list(self.maplayers.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['mapsty'] == 'fixed':
            pass
        elif self.hparams['mapsty'] == 'itera':
            self.optimizer1 = torch.optim.Adam(
                self.maplayers.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        else:
            raise NotImplementedError('Please provide a valid mapsty.')

    def align_loss(self, fea, y):
        """
        Compute optimal transport-inspired loss with mapping layers and NLP anchors.
        """
        losses = torch.zeros(len(y)).to(y.device)
        for i in range(len(fea)):
            losses[i] = -F.cosine_similarity(self.maplayers[y[i]](fea[i]), self.nlpanchor[y[i]], dim=0)
        weights = F.softmax((-losses.detach() * self.hparams['temp'])).detach()
        return torch.sum(losses * weights), weights

    def update_cold(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_fea = self.featurizer(all_x)
        loss = F.cross_entropy(self.classifier(all_fea), all_y)
        self.optimizer.zero_grad()
        if self.hparams['mapsty'] == 'itera':
            self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Moving Average network
        self.update_ma()

        return {'loss': loss.item()}

    def update_maplayer(self, minibatches, unlabeled=None):
        """
        Update mapping layers with OT loss and weighted cross-entropy loss.
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_fea = self.featurizer(all_x)

        alnloss, weights = self.align_loss(all_fea, all_y)
        loss = alnloss * self.hparams['lambda'] + torch.sum(
            F.cross_entropy(self.classifier(all_fea), all_y, reduction='none') * weights
        )
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()

        # Update Moving Average network
        self.update_ma()

        return {'loss': loss.item()}

    def update(self, minibatches, unlabeled=None):
        """
        Update the main model.
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_fea = self.featurizer(all_x)

        alnloss, weights = self.align_loss(all_fea, all_y)
        loss = alnloss * self.hparams['lambda'] + torch.sum(
            F.cross_entropy(self.classifier(all_fea), all_y, reduction='none') * weights
        )
        if self.hparams['mapsty'] == 'itera':
            self.optimizer1.zero_grad()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Moving Average network
        self.update_ma()

        return {'loss': loss.item()}

    def predict(self, x):
        self.network_ma.eval()
        return self.network_ma(x)

    def update_ma(self):
        """
        Update Moving Average (MA) model parameters.
        """
        self.global_iter += 1
        if self.global_iter >= self.ma_start_iter:
            self.ma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = (param_k.data * self.ma_count + param_q.data) / (1. + self.ma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = param_q.data
              
class NLPGERM_NoWeight(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(NLPGERM_NoWeight, self).__init__(input_shape, num_classes, num_domains,
                                      hparams)
        self.num_classes = num_classes

        # Initialize Moving Average (MA) network
        self.network_ma = copy.deepcopy(self.network)
        self.network_ma.eval()
        self.ma_start_iter = 100
        self.global_iter = 0
        self.ma_count = 0

    def set_nlp_anchor(self, dataset):
        """
        Sets NLP anchors using CLIP and averages multiple text prompts per class.
        """
        classes_names = dataset.datasets[0].classes
        import clip
        device = next(self.network.parameters()).device
        model, preprocess = clip.load("RN50", device)

        # Generate multiple prompts per class
        text_prompts = [
            [
                f"a photo of a {item}"
            ]
            for item in classes_names
        ]

        all_text_features = []

        for prompts in text_prompts:
            # Tokenize all prompts for the class
            text_tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                # Encode each prompt and average the embeddings
                text_features = model.encode_text(text_tokens)
                text_features_avg = text_features.mean(dim=0)  # Average the features
            all_text_features.append(text_features_avg)

        # Stack the averaged features to form the anchor matrix
        self.nlpanchor = torch.stack(all_text_features)

        # Define mapping layers
        self.maplayers = nn.ModuleList([
            nn.Linear(self.featurizer.n_outputs, len(self.nlpanchor[0])).to(device)
            for _ in range(self.num_classes)
        ])

        # Optimizer setup
        if self.hparams['mapsty'] == 'all':
            self.optimizer = torch.optim.Adam(
                list(self.network.parameters()) + list(self.maplayers.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['mapsty'] == 'fixed':
            pass
        elif self.hparams['mapsty'] == 'itera':
            self.optimizer1 = torch.optim.Adam(
                self.maplayers.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        else:
            raise NotImplementedError('Please provide a valid mapsty.')

    def align_loss(self, fea, y):
        """
        Compute optimal transport-inspired loss with mapping layers and NLP anchors.
        """
        losses = torch.zeros(len(y)).to(y.device)
        for i in range(len(fea)):
            losses[i] = -F.cosine_similarity(self.maplayers[y[i]](fea[i]), self.nlpanchor[y[i]], dim=0)
        # weights = F.softmax((-losses.detach() * 10)).detach()
        return torch.sum(losses)

    def update_cold(self, minibatches, unlabeled=None):
        """
        Update only the classifier with cross-entropy loss.
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_fea = self.featurizer(all_x)
        loss = F.cross_entropy(self.classifier(all_fea), all_y)
        self.optimizer.zero_grad()
        if self.hparams['mapsty'] == 'itera':
            self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Moving Average network
        self.update_ma()

        return {'loss': loss.item()}

    def update_maplayer(self, minibatches, unlabeled=None):
        """
        Update mapping layers with OT loss and weighted cross-entropy loss.
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_fea = self.featurizer(all_x)

        alnloss = self.align_loss(all_fea, all_y)
        loss = alnloss * self.hparams['lambda'] + torch.sum(
            F.cross_entropy(self.classifier(all_fea), all_y, reduction='none')
        )
        self.optimizer.zero_grad()
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()

        # Update Moving Average network
        self.update_ma()

        return {'loss': loss.item()}

    def update(self, minibatches, unlabeled=None):
        """
        Update the main model.
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_fea = self.featurizer(all_x)

        alnloss= self.align_loss(all_fea, all_y)
        loss = alnloss * self.hparams['lambda'] + torch.sum(
            F.cross_entropy(self.classifier(all_fea), all_y, reduction='none')
        )
        if self.hparams['mapsty'] == 'itera':
            self.optimizer1.zero_grad()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Moving Average network
        self.update_ma()

        return {'loss': loss.item()}

    def predict(self, x):
        self.network_ma.eval()
        return self.network_ma(x)

    def update_ma(self):
        """
        Update Moving Average (MA) model parameters.
        """
        self.global_iter += 1
        if self.global_iter >= self.ma_start_iter:
            self.ma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = (param_k.data * self.ma_count + param_q.data) / (1. + self.ma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = param_q.data

class OT(NLPGERM):
    """
    Two version of align_loss, one for EOT and one for weak OT
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains
        self.ot_reg = hparams['ot_reg']
        self.seed = hparams.get('random_seed', 0)
        self.ot_unbalanced = None


    @staticmethod
    def _concat_with_env(minibatches):
        """
        minibatches: list of (x, y) for each env
        return:
            all_x: (B,...)
            all_y: (B,)
            all_env: (B,) domain id = env_idx for every sample
        """
        xs, ys, envs = [], [], []
        for env_i, (x, y) in enumerate(minibatches):
            xs.append(x)
            ys.append(y)
            envs.append(torch.full_like(y, env_i))  # (B_env,)
        all_x = torch.cat(xs)
        all_y = torch.cat(ys)
        all_env = torch.cat(envs)
        return all_x, all_y, all_env

    def set_nlp_anchor(self, dataset, data_name='PACS'):
        """
        self.nlpanchor shape: (C, D, E)
        """
        import clip
        device = next(self.network.parameters()).device
        model, preprocess = clip.load("RN50", device)

        class_names = dataset.datasets[0].classes
        C = len(class_names)

        if hasattr(dataset.datasets[0], "env_name"):
            env_names = [d.env_name for d in dataset.datasets]
        else:
            env_names = ["art_painting", "cartoon", "photo", "sketch"]

        D = len(env_names)

        domain_prompt_prefix = {
            "art_painting": "a painting of a",
            "cartoon": "a cartoon of a",
            "photo": "a photo of a",
            "sketch": "a sketch of a",
        }

        anchors = []

        for cls_name in class_names:            # C
            cls_anchors = []
            for env_name in env_names:          # D
                prefix = domain_prompt_prefix.get(env_name, "a photo of a")
                prompt = f"{prefix} {cls_name}"
                text_tokens = clip.tokenize([prompt]).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens) # (1, E)
                    text_features = text_features[0].float()
                cls_anchors.append(text_features)
            # (D, E)
            cls_anchors = torch.stack(cls_anchors, dim=0)
            anchors.append(cls_anchors)

        # (C, D, E)
        anchors = torch.stack(anchors, dim=0).float() # (C, D, E)
        anchors = F.normalize(anchors, dim=-1)

        self.nlpanchor = anchors.to(device)   # (num_classes, num_domains, dim_embed)
        _, _, dim_embed = self.nlpanchor.shape

        self.maplayers = nn.ModuleList([
            nn.Linear(self.featurizer.n_outputs, dim_embed).to(device)
            for _ in range(self.num_classes)
        ])

        if self.hparams['mapsty'] == 'all':
            self.optimizer = torch.optim.Adam(
                list(self.network.parameters()) + list(self.maplayers.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['mapsty'] == 'fixed':
            pass
        elif self.hparams['mapsty'] == 'itera':
            self.optimizer1 = torch.optim.Adam(
                self.maplayers.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        else:
            raise NotImplementedError('Please provide a valid mapsty.')
    # ------- EOT -------

    def align_loss(self, fea, y, env):
        """
        Args:
            fea: Tensor of shape (batch(B), dim_feat)
            y: (batch(B),) class id
            env: (batch(B),) domain id

        Requires:
            self.nlpanchor: Tensor of shape (n_classes, n_domains, dim_embed)
                            self.nlpanchor[c, d] is the embedding vector for class c, domain d
            self.maplayers: nn.ModuleList of length n_classes
                            where self.maplayers[c] maps feat-space -> embed-space
                            
        Returns:
            loss: scalar tensor
            weights: tensor of shape (batch,)
        """
        device = fea.device
        B = fea.size(0)
        losses = torch.zeros(B, device=device, dtype=fea.dtype)

        classes = y
        domains = env
        # Unique classes in this batch
        unique_classes = classes.unique()

        for cls in unique_classes:
            # Indices of samples belonging to this class
            cls_int = int(cls.item())

            mask = (classes == cls)
            idx = mask.nonzero(as_tuple=False).squeeze(1)

            # --- Collect the features & anchors for this class ---
            feat_cls = fea[idx] # (n_cls, dim_feat)
            dom_cls  = domains[idx] # (n_cls,)

            # Map features into embedding space using the class-specific projector
            mapped_feat_cls = self.maplayers[cls_int](feat_cls) # (n_cls, dim_embed)
            mapped_feat_cls = F.normalize(mapped_feat_cls, dim=-1)

            # Pick anchors for the same (class, domain) pairs
            # self.nlpanchor[cls] is (n_domains, dim_embed), index by domain ids
            nlp_anchor_cls = self.nlpanchor[cls_int, dom_cls] # (n_cls, dim_embed)

            # Here we assume solve_sample returns an object with `.value` (scalar)
            # You may want to pass weights a,b or cost matrix if your API needs it.
            # Distribute this class loss over all samples in the class
            ot_res = ot.solve_sample(
                mapped_feat_cls,
                nlp_anchor_cls,
                grad='envelope',
                random_state=self.seed,
                reg=self.ot_reg,
                unbalanced=self.ot_unbalanced,
            )
            # if torch.isnan(ot_res.value):
            #     print("OT value is NaN for class", cls_int)
            losses[idx] = ot_res.value
        if not hasattr(self, "debug_cnt"):
            self.debug_cnt = 0
        # self.debug_cnt += 1
        # if self.debug_cnt <= 5 or self.debug_cnt % 50 == 0:
        #     print("align_loss: mean =", losses.mean().item(),
        #         "min =", losses.min().item(),
        #         "max =", losses.max().item())

        # Soft weighting across samples as in your original code
        # Note: sign convention – we treat larger cost as "worse", so keep it positive.
        weights = F.softmax(-losses.detach() * self.hparams['temp'], dim=0).detach()
        total_loss = torch.sum(losses * weights)

        return total_loss, weights
    # ------- Weak OT -------
    def align_loss(self, fea, y, env):
        """Weak Optimal Transport based NLP-GERM.

        This class reuses all infrastructure from `OT` (anchors, maplayers, etc.)
        but overrides `align_loss` to use POT's *weak* OT solver.

        Differentiation follows the envelope theorem:

            - Inner problem:  γ* = argmin_γ  Σ_i a_i || x_i - (1/a_i Σ_j γ_ij y_j ) ||^2
            - Outer loss:     F(x, y; γ*)    = Σ_i a_i || x_i - (1/a_i Σ_j γ*_ij y_j ) ||^2

        We first solve for γ* with POT on CPU (no gradients).
        Then we *re-evaluate* the weak-OT objective F in PyTorch using γ* as a
        constant. Gradients flow only through x (mapped features) and y
        (anchors, if they were learnable), but not through γ*.
        """

        device = fea.device
        dtype  = fea.dtype
        B      = fea.size(0)

        classes = y # (B,)
        domains = env # (B,)

        losses = torch.zeros(B, device=device, dtype=dtype)

        unique_classes = classes.unique()
        for cls in unique_classes:
            cls_int = int(cls.item())

            mask = (classes == cls)
            idx  = mask.nonzero(as_tuple=False).squeeze(1) # (ns,)
            ns   = idx.numel()
            if ns == 0:
                continue
            feat_cls = fea[idx] # (ns, dim_feat)
            dom_cls  = domains[idx] # (ns,)

            mapped_feat_cls = self.maplayers[cls_int](feat_cls) # (ns, dim_embed)
            mapped_feat_cls = F.normalize(mapped_feat_cls, dim=-1)

            nlp_anchor_cls = self.nlpanchor[cls_int, dom_cls] # (ns, dim_embed)
            # nlp_anchor_cls = F.normalize(nlp_anchor_cls, dim=-1)

            '''--------------- Weak OT specific code below ----------------'''
            # ------- Prepare data for POT (CPU, numpy, detached) -------
            Xa_np = mapped_feat_cls.detach().cpu().numpy().astype(np.float64) # (ns, E)
            Xb_np = nlp_anchor_cls.detach().cpu().numpy().astype(np.float64) # (ns, E)

            # ------- Solve weak OT on CPU (no gradients) -------
            # gamma_np has shape (ns, ns). This is γ* in the envelope theorem.
            gamma_np = ot.weak_optimal_transport(Xa_np, Xb_np)
            # Bring γ* back to torch *without* connecting it to the autograd graph.
            gamma = torch.as_tensor(gamma_np, device=device, dtype=dtype) # (ns, ns)

            # ------- Re-evaluate weak-OT objective in PyTorch -------
            # a_distr is the uniform distribution over source samples
            a_distr = torch.full((ns,), 1.0 / ns, device=device, dtype=dtype)  # (ns,)

            # Barycentric projection of targets:
            # bary_i = (1 / a_i) * Σ_j γ_ij * Xb_j
            # gamma: (ns, ns), nlp_anchor_class: (ns, dim_embed)
            # => (ns, dim_embed)
            bary = (gamma @ nlp_anchor_cls) / a_distr.unsqueeze(1).clamp_min(1e-12)  # (ns, E)

            # Per-sample weak-OT cost: a_i * ||x_i - bary_i||^2
            cost_per_sample = a_distr * (mapped_feat_cls - bary).pow(2).sum(dim=1)   # (ns,)

            # Store per-sample costs back into the global losses tensor
            losses[idx] = cost_per_sample

        # Soft weighting across samples as in your original code
        # Note: sign convention – we treat larger cost as "worse", so keep it positive.
        weights = F.softmax(-losses.detach() * self.hparams['temp'], dim=0).detach()
        total_loss = torch.sum(losses * weights)

        return total_loss, weights

    def update(self, minibatches, unlabeled=None):
        """
        NLPGERM update + env
        """
        all_x, all_y, all_env = self._concat_with_env(minibatches)
        if unlabeled is not None:
            all_env = unlabeled
        else:
            _, _, all_env = self._concat_with_env(minibatches)
        if self.global_iter < 5:
            print("[OT.update] all_env unique:", all_env.unique().tolist())
        all_fea = self.featurizer(all_x)

        alnloss, weights = self.align_loss(all_fea, all_y, all_env)
        loss = alnloss * self.hparams['lambda'] + torch.sum(
            F.cross_entropy(self.classifier(all_fea), all_y, reduction='none') * weights
        )

        if self.hparams['mapsty'] == 'itera':
            self.optimizer1.zero_grad()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_ma()

        return {'loss': loss.item()}