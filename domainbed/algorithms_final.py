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
        classes_names = ['bird', 'car', 'chair', 'dog', 'person']
        
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

    # TODO: change cosine similarity to EOT begin
    def align_loss(self, fea, y):
        device = fea.device
        dtype  = fea.dtype
        eps    = self.hparams.get('ot_eps',   0.05)
        iters  = self.hparams.get('ot_iters', 50)
        T      = self.hparams.get('temp',     1.0)

        B = fea.size(0)
        D = self.nlpanchor.size(1)

        mapped = torch.empty(B, D, device=device, dtype=dtype)
        anchor = torch.empty(B, D, device=device, dtype=dtype)

        for c in range(self.num_classes):
            idx = (y == c).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            mapped[idx] = self.maplayers[c](fea[idx])

            anchor_c = self.nlpanchor[c].to(device=device, dtype=dtype)  # (D,)
            anchor[idx] = anchor_c.unsqueeze(0).expand(idx.numel(), -1)

        a = F.softmax(mapped, dim=1)
        b = F.softmax(anchor, dim=1)
        a = a / a.sum(dim=1, keepdim=True)
        b = b / b.sum(dim=1, keepdim=True)

        idx = torch.arange(D, device=device, dtype=dtype)
        M = (idx[:, None] - idx[None, :]) ** 2          # (D, D)
        M_batch = M.unsqueeze(0).expand(B, D, D).contiguous()

        res_ab = ot.solve_batch(
            M=M_batch, a=a, b=b,
            reg=eps, max_iter=iters, grad="envelope"
        )
        res_aa = ot.solve_batch(
            M=M_batch, a=a, b=a,
            reg=eps, max_iter=iters, grad="envelope"
        )
        res_bb = ot.solve_batch(
            M=M_batch, a=b, b=b,
            reg=eps, max_iter=iters, grad="envelope"
        )

        ot_ab = res_ab.value          # (B,)
        ot_aa = res_aa.value
        ot_bb = res_bb.value

        losses = 2 * ot_ab - ot_aa - ot_bb          # (B,)
        weights = F.softmax(-losses.detach() * T, dim=0).detach()
        loss = (losses * weights).sum()
        if torch.rand(1).item() < 0.001:
            print(f"[DEBUG align_loss] min={float(losses.min()):.4f}, "
                f"max={float(losses.max()):.4f}, mean={float(losses.mean()):.4f}, "
                f"w_min={float(weights.min()):.4f}, w_max={float(weights.max()):.4f}")

        return loss, weights

    # def align_loss(self, fea, y):
    #     """
    #     Compute optimal transport-inspired loss with mapping layers and NLP anchors.
    #     """
    #     losses = torch.zeros(len(y)).to(y.device)
    #     for i in range(len(fea)):
    #         losses[i] = -F.cosine_similarity(self.maplayers[y[i]](fea[i]), self.nlpanchor[y[i]], dim=0)
    #     weights = F.softmax((-losses.detach() * self.hparams['temp'])).detach()
    #     return torch.sum(losses * weights), weights

    # TODO: change cosine similarity to EOT end

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
        ce_per  = F.cross_entropy(self.classifier(all_fea), all_y, reduction='none')
        ce_loss = (ce_per * weights).sum()
        loss = alnloss * self.hparams['lambda'] + ce_loss
        if torch.rand(1).item() < 0.001:
            print(
                f"[DEBUG loss] aln={float(alnloss):.4f}, "
                f"ce={float(ce_loss):.4f}, "
                f"λ={self.hparams['lambda']}, "
                f"T={self.hparams.get('temp', 1.0)}"
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
        classes_names = ['bird', 'car', 'chair', 'dog', 'person']
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
