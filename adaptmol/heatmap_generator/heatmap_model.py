from numpy import recfromcsv
import torch
import torch.nn as nn
import torch.nn.functional as F



class SequenceHeatmapGenerator(nn.Module):
    """
    Generates spatial heatmaps from sequence logits by modeling joint distributions
    of x/y coordinate tokens.

    Args:
        vocab_size (int): Total vocabulary size.
        target_height (int): Output heatmap height in pixels. Default 256.
        target_width (int): Output heatmap width in pixels. Default 256.
        x_token_start (int): Vocab index where x-coordinate tokens begin. Default 101.
        x_token_end (int): Vocab index where x-coordinate tokens end. Default 165.
        y_token_start (int): Vocab index where y-coordinate tokens begin. Default 165.
        start_token (int): Index of the sequence start token. Default 1.
        aggregation (str): Heatmap aggregation strategy, one of
                           'direct_pairs', 'weighted_pairs', or 'attention'. Default 'direct_pairs'.
        significance_threshold (float): Threshold for filtering low-confidence predictions. Default 0.3.

    Forward Args:
        seq_logits (Tensor): Shape (B, T, V) — batch of sequence logits.
        return_details (bool): If True, also return a dict with intermediate outputs. Default False.

    Forward Returns:
        Tensor: Heatmap of shape (B, 1, target_height, target_width).
        
    """
    
    def __init__(self, vocab_size, target_height=256, target_width=256, 
                 x_token_start=101, x_token_end=165, y_token_start=165, 
                 start_token=1, aggregation='direct_pairs', significance_threshold=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_height = target_height
        self.target_width = target_width
        self.x_token_start = x_token_start
        self.x_token_end = x_token_end  
        self.y_token_start = y_token_start
        self.start_token = start_token
        self.aggregation = aggregation
        self.significance_threshold = significance_threshold
        
       
        self.x_range = x_token_end - x_token_start  
        self.y_range = vocab_size - y_token_start
        
  
        self.post_process = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
            
        )
        
       
        if aggregation == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(vocab_size, 64),
                nn.ReLU(), 
                nn.Linear(64, 1)
                
            )
        
    def forward(self, seq_logits, return_details=False):
       
        B, T, V = seq_logits.size()
        device = seq_logits.device
        
       
        probs = F.softmax(seq_logits, dim=-1)  
        
       
        x_probs = probs[:, :, self.x_token_start:self.x_token_end]  # (B, T, 64)
        y_probs = probs[:, :, self.y_token_start:]  # (B, T, y_range)
        
        if T < 2:
           
            return torch.zeros(B, 1, self.target_height, self.target_width, device=device), torch.tensor(0.0)
        
       
        if self.aggregation == 'direct_pairs':
            heatmap = self._generate_heatmap_from_pairs(x_probs, y_probs, probs)
        elif self.aggregation == 'weighted_pairs':
            heatmap = self._generate_weighted_heatmap(x_probs, y_probs, probs)
        elif self.aggregation == 'attention':
            heatmap = self._generate_attention_heatmap(x_probs, y_probs, probs)
        else:
            
            heatmap = self._generate_heatmap_from_pairs(x_probs, y_probs, probs)
        
       
        heatmap = self.post_process(heatmap)
        
        
        
        if return_details:
            details = {
                'x_probs': x_probs,
                'y_probs': y_probs,
                'raw_heatmap': heatmap,
                'method': self.aggregation
            }
            return heatmap, details
        
        return heatmap
    
    def _generate_heatmap_from_pairs(self, x_probs, y_probs, probs):
        """
        Generate heatmap from joint distribution of consecutive x/y token pairs.

        Args:
            x_probs (Tensor): Shape (B, T, x_range).
            y_probs (Tensor): Shape (B, T, y_range).
            probs (Tensor): Full vocab probs (B, T, V), reserved for future use.

        Returns:
            Tensor: Heatmap of shape (B, 1, target_height, target_width).
        """
        
        B, T, _ = x_probs.shape
        device = x_probs.device
        
       
        x_pairs = x_probs[:, :-1, :]    
        y_pairs = y_probs[:, 1:, :]     
        
        
        joint_distribution = torch.einsum('bti,btj->bij', y_pairs, x_pairs)  
        
        
        heatmap = F.interpolate(
            joint_distribution.unsqueeze(1),  # (B, 1, y_range, 64)
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=True
        )  
        
        return heatmap
    
    def _generate_weighted_heatmap(self, x_probs, y_probs, probs):
       
        B, T, _ = x_probs.shape
        device = x_probs.device
        
        
        x_prob_mass = torch.sum(x_probs, dim=-1)  # (B, T)
        y_prob_mass = torch.sum(y_probs, dim=-1)  # (B, T)
        max_prob_per_step = torch.max(probs, dim=-1)[0]  # (B, T)
        
        
        x_is_significant = x_prob_mass > (max_prob_per_step * self.significance_threshold)
        y_is_significant = y_prob_mass > (max_prob_per_step * self.significance_threshold)
        
        
        valid_xy_pairs = x_is_significant[:, :-1] & y_is_significant[:, 1:]  # (B, T-1)
        
        
        x_pairs = x_probs[:, :-1, :]    # (B, T-1, 64)
        y_pairs = y_probs[:, 1:, :]     # (B, T-1, y_range)
        
        
        valid_weights = valid_xy_pairs.float().unsqueeze(-1)  # (B, T-1, 1)
        x_pairs_weighted = x_pairs * valid_weights
        y_pairs_weighted = y_pairs * valid_weights
        
       
        joint_distribution = torch.einsum('bti,btj->bij', y_pairs_weighted, x_pairs_weighted)
        
        
        heatmap = F.interpolate(
            joint_distribution.unsqueeze(1),
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=True
        )
        
        return heatmap
    
    def _generate_attention_heatmap(self, x_probs, y_probs, probs):
        
        B, T, _ = x_probs.shape
        
        
        attn_weights = self.temporal_attention(probs) 
        attn_weights = F.softmax(attn_weights, dim=1)
        
        
        x_probs_agg = torch.sum(x_probs * attn_weights, dim=1)  # (B, 64)
        y_probs_agg = torch.sum(y_probs * attn_weights, dim=1)  # (B, y_range)
        
        
        joint_distribution = torch.bmm(
            y_probs_agg.unsqueeze(-1),    # (B, y_range, 1)
            x_probs_agg.unsqueeze(1)      # (B, 1, 64)
        )  # (B, y_range, 64)
        
        
        heatmap = F.interpolate(
            joint_distribution.unsqueeze(1),
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=True
        )
        
        return heatmap
    














       


