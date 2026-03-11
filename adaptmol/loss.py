import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .tokenizer import PAD_ID, MASK, MASK_ID


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # assuming output is raw logits
        # convert to log_probs
        log_probs = F.log_softmax(output, dim=-1)

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # reduction mean or sum?
        return F.kl_div(log_probs, model_prob, reduction='batchmean')


class SequenceLoss(nn.Module):
    """
    Cross-entropy or label-smoothing loss for sequence prediction.

    Args:
        label_smoothing (float): Smoothing factor; 0 disables smoothing.
        vocab_size (int): Target vocabulary size.
        ignore_index (int): Token index to ignore in loss. Default -100.
        ignore_indices (list[int]): Additional indices to mask out. Default [].
    """
    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        if ignore_indices:
            ignore_index = ignore_indices[0]
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index)
        
    def forward(self, output, target):
        """
        :param output: [batch, len, vocab]
        :param target: [batch, len]
        :return:
        """
        batch_size, max_len, vocab_size = output.size()
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill_((target == idx), self.ignore_index)
        loss = self.criterion(output, target)
        return loss




class GraphLoss(nn.Module):
    """
    Weighted cross-entropy loss for bond type (edge) prediction,
    with optional MMD regularization between labeled and unlabeled outputs.
    """
    def __init__(self):
        super(GraphLoss, self).__init__()
        weight = torch.ones(7) * 10
        weight[0] = 5
        weight[5] = 25
        weight[6] = 25
        weight[2] = 20
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)
        
       
        self.mmd_weight = 0.1  
        self.epoch = 0
    def compute_mmd(self, outputs, targets, outputs_unlabeled=None):
        """
        Compute MMD regularization loss between labeled and unlabeled predictions.

        Args:
            outputs (dict): Labeled model outputs containing 'edges'.
            targets (dict): Ground-truth labels containing 'edges'.
            outputs_unlabeled (dict, optional): Unlabeled model outputs. Default None.

        Returns:
            dict: {'mmd': loss_tensor} if MMD is computed, else {}.
        """
        
        results = {}
        if outputs_unlabeled is not None:
            mmd_loss = self._compute_mmd_loss(outputs, targets, outputs_unlabeled)
            if mmd_loss is not None:
                if self.epoch == 0:
                    weight = self.mmd_weight
                elif self.epoch == 1:
                    weight = self.mmd_weight * 0.75
                else:
                    weight = self.mmd_weight * 0.5
                results['mmd'] = weight * mmd_loss
        self.epoch += 1
        return results

    def forward(self, outputs, targets, outputs_unlabeled=None):
        
        results = {}
        
        if 'coords' in outputs:
            pred = outputs['coords']
            max_len = pred.size(1)
            target = targets['coords'][:, :max_len]
            mask = target.ge(0)
            loss = F.l1_loss(pred, target, reduction='none')
            results['coords'] = (loss * mask).sum() / mask.sum()
            
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)
        
        return results

    def _compute_mmd_loss(self, gt_outputs, gt_targets, pred_outputs, ignore_index=-100):
        """
        Compute per-class MMD loss between GT and unlabeled edge feature distributions.

        Returns:
            Tensor | None: Scalar MMD loss, or None if insufficient samples.
        """
       
        if ('edges' not in gt_outputs) or ('edges' not in pred_outputs) or ('edges' not in pred_outputs['edges']):
            return None
        if ('edges' not in gt_targets):
            return None

       
        gt_edges = gt_outputs['edges']  # (B, 7, N, N)
        gt_probs = torch.softmax(gt_edges, dim=1)
        gt_pred_labels = gt_probs.argmax(dim=1)
        gt_max_probs = gt_probs.max(dim=1).values
        gt_entropy = -(gt_probs * (gt_probs + 1e-8).log()).sum(dim=1)

        
        gt_labels = gt_targets['edges'].long()

       
        match_mask = (gt_pred_labels == gt_labels)
        if ignore_index is not None:
            match_mask = match_mask & (gt_labels != ignore_index)

        
        confidence_threshold = 0.95 
        if self.epoch > 0:
            confidence_threshold = 0.98
        entropy_threshold = 0.1
        if self.epoch > 0:
            entropy_threshold = 0.05
        

        pred_edges = pred_outputs['edges']['edges']
        pred_probs = torch.softmax(pred_edges, dim=1)
        pred_pred_labels = pred_probs.argmax(dim=1)
        pred_max_probs = pred_probs.max(dim=1).values
        pred_entropy = -(pred_probs * (pred_probs + 1e-8).log()).sum(dim=1)

        gt_reliable_mask = (gt_max_probs > confidence_threshold) & (gt_entropy < entropy_threshold)
        gt_mask_final = match_mask & gt_reliable_mask

        pred_reliable_mask = (pred_max_probs > confidence_threshold) & (pred_entropy < entropy_threshold)

        if gt_mask_final.sum() < 50 or pred_reliable_mask.sum() < 50:
            return None



        gt_edge_features = gt_outputs.get('_atom_pairs', None)  # (B, N, N, 256)
        pred_edge_features = pred_outputs['edges'].get('_atom_pairs', None)
        
        if (gt_edge_features is None) or (pred_edge_features is None):
            return None


        with torch.no_grad():

            feat_dim = gt_edge_features.shape[-1]
            g_sample = gt_edge_features.reshape(-1, feat_dim)[:1000]
            p_sample = pred_edge_features.reshape(-1, feat_dim)[:1000]
            all_samples = torch.cat([g_sample, p_sample], dim=0)
            

            feat_std = all_samples.std(dim=0, keepdim=True).clamp(min=0.1)


        gt_scaled = gt_edge_features / feat_std.view(1, 1, 1, -1)
        pred_scaled = pred_edge_features / feat_std.view(1, 1, 1, -1)


        mmd_losses = []
        valid_classes = []

        for class_id in range(5):  

            gt_mask_cls = gt_mask_final & (gt_labels == class_id)
            

            gt_features = gt_scaled[gt_mask_cls]

            if gt_features.size(0) > 50:  
                gt_conf_sel = gt_max_probs[gt_mask_cls]
                _, idx_top = gt_conf_sel.topk(50)
                gt_features = gt_features[idx_top]

            pred_mask_cls = (pred_pred_labels == class_id) & pred_reliable_mask
           

            pred_features = pred_scaled[pred_mask_cls]
            if pred_features.size(0) > 50:
                pred_conf_sel = pred_max_probs[pred_mask_cls]
                _, idx_top = pred_conf_sel.topk(50)
                pred_features = pred_features[idx_top]
            

            mmd_val = self._stable_mmd_for_gelu(gt_features, pred_features, class_id)
            if (mmd_val is not None) and (mmd_val > 0):
                mmd_losses.append(mmd_val)
                valid_classes.append(class_id)


        if len(mmd_losses) >= 2:
            final_mmd = torch.stack(mmd_losses).mean()
            final_mmd = torch.clamp(final_mmd, min=0.0, max=5.0)  
            return final_mmd
        else:
            return None

    def _stable_mmd_for_gelu(self, x, y, class_id):
        """
        Compute MMD between two feature sets using multiple RBF kernels.

        Args:
            x (Tensor): GT features of shape (N, D).
            y (Tensor): Predicted features of shape (M, D).
            class_id (int): Bond class id, used to select kernel bandwidths.

        Returns:
            Tensor | None: Scalar MMD value, or None if computation is unstable.
        """
        n = min(50, x.size(0), y.size(0))
        if x.size(0) > n:
            x = x[:n]
        if y.size(0) > n:
            y = y[:n]
        

        if class_id == 0:  # no bond
            sigmas = [0.25, 0.5, 1.0]  
        elif class_id in [5, 6]: 
            sigmas = [0.5, 1.0, 1.5]
        else:  
            sigmas = [0.3, 0.6, 1.2]
        
        mmd_values = []
        
        for sigma in sigmas:
            
            dist_xx = torch.cdist(x, x, p=2)
            dist_yy = torch.cdist(y, y, p=2)
            dist_xy = torch.cdist(x, y, p=2)
            
           
            max_dist = 5.0  
            dist_xx = torch.clamp(dist_xx, max=max_dist)
            dist_yy = torch.clamp(dist_yy, max=max_dist)
            dist_xy = torch.clamp(dist_xy, max=max_dist)
            
          
            kxx = torch.exp(-dist_xx**2 / (2 * sigma**2 + 1e-8))
            kyy = torch.exp(-dist_yy**2 / (2 * sigma**2 + 1e-8))
            kxy = torch.exp(-dist_xy**2 / (2 * sigma**2 + 1e-8))
            
           
            n_x = x.size(0)
            n_y = y.size(0)
            
            if n_x > 1:
                kxx = (kxx.sum() - kxx.trace()) / (n_x * (n_x - 1))
            else:
                kxx = torch.tensor(0.0, device=x.device)
                
            if n_y > 1:
                kyy = (kyy.sum() - kyy.trace()) / (n_y * (n_y - 1))
            else:
                kyy = torch.tensor(0.0, device=y.device)
                
            kxy = kxy.mean()
            
            # MMD
            mmd = kxx + kyy - 2 * kxy
            mmd = torch.abs(mmd)  
            
            if not torch.isnan(mmd) and not torch.isinf(mmd) and mmd < 10:
                mmd_values.append(mmd)
        
        if len(mmd_values) > 0:
            return torch.stack(mmd_values).mean()
        else:
            return None







class Criterion(nn.Module):
    """
    Composite loss module combining per-format sequence and graph losses.

    Args:
        args: Training arguments with 'formats' and 'label_smoothing'.
        tokenizer (dict): Format-keyed tokenizer instances.
    """
    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}
        for format_ in args.formats:
            if format_ == 'edges':
                criterion['edges'] = GraphLoss()
            else:
                if MASK in tokenizer[format_].stoi:
                    ignore_indices = [PAD_ID, MASK_ID]
                else:
                    ignore_indices = []
                criterion[format_] = SequenceLoss(args.label_smoothing, len(tokenizer[format_]),
                                                  ignore_index=PAD_ID, ignore_indices=ignore_indices)
        self.criterion = nn.ModuleDict(criterion)


    def forward(self, results, refs, gt_heatmap, result_mmd = None):
        losses = {}
        

        if result_mmd is not None:
            format_ = "edges"
            predictions, targets, *_ = results[format_]
            loss_ = self.criterion[format_].compute_mmd(predictions, targets,result_mmd)
            if type(loss_) is dict:
                losses.update(loss_)
            else:
                if loss_.numel() > 1:
                    loss_ = loss_.mean()
                losses["mmd"] = loss_
            return losses


        for format_ in results:
            if format_ == 'chartok_coords':
                predictions, targets, _, seq_heatmap,_  = results[format_]
                # print(seq_heatmap.shape, hidden_heatmap.shape, gt_heatmap.shape)
                pos_weight = 50.0
                weight = torch.where(gt_heatmap > 0, pos_weight , 1.0)


                mse_error2 = (seq_heatmap - gt_heatmap) ** 2  
                gt_heatmap_loss = (weight * mse_error2).sum() / weight.sum()
                loss_ = self.criterion[format_](predictions, targets)
                if type(loss_) is dict:
                    losses.update(loss_)
                else:
                    if loss_.numel() > 1:
                        loss_ = loss_.mean()
                    losses[format_] = loss_
                losses['gt_heatmap'] = gt_heatmap_loss
            else:
                predictions, targets, *_ = results[format_]
                loss_ = self.criterion[format_](predictions, targets)
                if type(loss_) is dict:
                    losses.update(loss_)
                else:
                    if loss_.numel() > 1:
                        loss_ = loss_.mean()
                    losses[format_] = loss_
        return losses
