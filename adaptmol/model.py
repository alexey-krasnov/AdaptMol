import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torch.nn.utils.rnn import pad_sequence
from .utils import FORMAT_INFO, to_device
from .tokenizer import SOS_ID, EOS_ID, PAD_ID, MASK_ID
from .inference import GreedySearch, BeamSearch
from .transformer import TransformerDecoder, Embeddings
from .heatmap_generator.heatmap_model import SequenceHeatmapGenerator

class Encoder(nn.Module):
    """
    Image encoder supporting ResNet, EfficientNet, and Swin Transformer backbones.

    Args:
        args: Arguments with 'encoder' (model name) and 'use_checkpoint' fields.
        pretrained (bool): If True, load pretrained weights. Default False.
    """
    def __init__(self, args, pretrained=False):
        super().__init__()
        model_name = args.encoder
        self.model_name = model_name
        if model_name.startswith('resnet'):
            self.model_type = 'resnet'
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.num_features  # encoder_dim
            self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('swin'):
            self.model_type = 'swin'
            self.transformer = timm.create_model(model_name, pretrained=pretrained, pretrained_strict=False,
                                                 use_checkpoint=args.use_checkpoint)
            self.n_features = self.transformer.num_features
            self.transformer.head = nn.Identity()
        elif 'efficientnet' in model_name:
            self.model_type = 'efficientnet'
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.num_features
            self.cnn.global_pool = nn.Identity()
            self.cnn.classifier = nn.Identity()
        else:
            raise NotImplemented

    def swin_forward(self, transformer, x):
        """
        Custom Swin Transformer forward pass that returns intermediate layer outputs.

        Args:
            transformer: Swin Transformer model instance.
            x (Tensor): Input image tensor.

        Returns:
            tuple[Tensor, list[Tensor]]: Final features and list of per-layer hidden states.
        """
        x = transformer.patch_embed(x)
        if transformer.absolute_pos_embed is not None:
            x = x + transformer.absolute_pos_embed
        x = transformer.pos_drop(x)

        def layer_forward(layer, x, hiddens):
            for blk in layer.blocks:
                if not torch.jit.is_scripting() and layer.use_checkpoint:
                    x = torch.utils.checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            H, W = layer.input_resolution
            B, L, C = x.shape
            hiddens.append(x.view(B, H, W, C))
            if layer.downsample is not None:
                x = layer.downsample(x)
            return x, hiddens

        hiddens = []
        for layer in transformer.layers:
            x, hiddens = layer_forward(layer, x, hiddens)
        x = transformer.norm(x)  # B L C
        hiddens[-1] = x.view_as(hiddens[-1])
        return x, hiddens

    def forward(self, x, refs=None):
        """
        Encode input images into spatial feature maps.

        Args:
            x (Tensor): Input image batch of shape (B, C, H, W).
            refs: Unused. Default None.

        Returns:
            tuple[Tensor, list[Tensor]]: Feature map (B, H, W, C) and list of hidden states.
        """
        if self.model_type in ['resnet', 'efficientnet']:
            features = self.cnn(x)
            features = features.permute(0, 2, 3, 1)
            hiddens = []
        elif self.model_type == 'swin':
            if 'patch' in self.model_name:
                features, hiddens = self.swin_forward(self.transformer, x)
            else:
                features, hiddens = self.transformer(x)
        else:
            raise NotImplemented
        return features, hiddens


class TransformerDecoderBase(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_trans_layer = nn.Sequential(
            nn.Linear(args.encoder_dim, args.dec_hidden_size)
            # nn.LayerNorm(args.dec_hidden_size, eps=1e-6)
        )
        self.enc_pos_emb = nn.Embedding(144, args.encoder_dim) if args.enc_pos_emb else None

        self.decoder = TransformerDecoder(
            num_layers=args.dec_num_layers,
            d_model=args.dec_hidden_size,
            heads=args.dec_attn_heads,
            d_ff=args.dec_hidden_size * 4,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=args.hidden_dropout,
            attention_dropout=args.attn_dropout,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=0,
            alignment_heads=0,
            pos_ffn_activation_fn='gelu'
        )

    def enc_transform(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        max_len = encoder_out.size(1)
        device = encoder_out.device
        if self.enc_pos_emb:
            pos_emb = self.enc_pos_emb(torch.arange(max_len, device=device)).unsqueeze(0)
            encoder_out = encoder_out + pos_emb
        #print(encoder_out.shape)
        encoder_out = self.enc_trans_layer(encoder_out)
        #print(encoder_out.shape)
        return encoder_out


class TransformerDecoderAR(TransformerDecoderBase):
    """Autoregressive Transformer Decoder"""

    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        print(self.vocab_size,"vocab_size")
        self.output_layer = nn.Linear(args.dec_hidden_size, self.vocab_size, bias=True)
        self.embeddings = Embeddings(
            word_vec_size=args.dec_hidden_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=args.hidden_dropout)
        
        self.seq_heatmap = SequenceHeatmapGenerator(self.vocab_size, 256, 256)
        self.teacher_forcing_ratio = 0.0
        self.dec_hidden_size = args.dec_hidden_size
    def dec_embedding(self, tgt, step=None):
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt.data.eq(pad_idx).transpose(1, 2)  # [B, 1, T_tgt]
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # batch x len x embedding_dim
        return emb, tgt_pad_mask

    
    def forward(self, encoder_out, labels, label_lengths, need_heatmap=False):
        """Training mode"""
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)

        tgt = labels.unsqueeze(-1)  # (b, t, 1)
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank, tgt_pad_mask=tgt_pad_mask)

        logits = self.output_layer(dec_out)  # (b, t, h) -> (b, t, v)
        if need_heatmap:
            seq_heatmapout = self.seq_heatmap(logits)
           
            
         
            return logits[:, :-1], labels[:, 1:], dec_out, seq_heatmapout,memory_bank
        return logits[:, :-1], labels[:, 1:], dec_out, None,memory_bank

        # return logits[:, :-1], labels[:, 1:], dec_out



    def inference(self, encoder_out, labels=None):
        """Inference mode. Autoregressively decode the sequence. Only greedy search is supported now. Beam search is
        out-dated. The labels is used for partial prediction, i.e. part of the sequence is given. In standard decoding,
        labels=None."""
        format_ = 'chartok_coords'
        max_len =  FORMAT_INFO[format_]['max_len']
        results = {}
        predictions = []
        results[format_] = self.decode(encoder_out, 1, 1, max_length=max_len)
        outputs, scores, token_scores, *_ = results[format_]
        beam_preds = [[self.tokenizer.sequence_to_smiles(x.tolist()) for x in pred]
                        for pred in outputs]
        predictions = [{format_: pred[0]} for pred in beam_preds]
        dec_out = results[format_][3]
        image_memory = results[format_][4]
        # print("=== dec_out ===")
        # print_shape_recursive(dec_out, "dec_out")

        # print("\n=== image_memory ===")
        # print_shape_recursive(image_memory, "image_memory")

        # print("\n=== predictions ===")
        # print_shape_recursive(predictions, "predictions")

        dec_out_tensors = [dec_out[i][0] for i in range(len(dec_out))]
        padded_dec_out = pad_sequence(dec_out_tensors, batch_first=True, padding_value=0.0)

        # ?? predictions indices: List[Dict] -> Tensor [b, ??]  
        indices_tensors = [torch.tensor(predictions[i]['chartok_coords']['indices'], dtype=torch.long) 
                        for i in range(len(predictions))]
        padded_indices = pad_sequence(indices_tensors, batch_first=True, padding_value=PAD_ID)

        return padded_dec_out, padded_indices, image_memory





    def decode(self, encoder_out, beam_size: int, n_best: int, min_length: int = 1, max_length: int = 256,
               labels=None):
        """Inference mode. Autoregressively decode the sequence. Only greedy search is supported now. Beam search is
        out-dated. The labels is used for partial prediction, i.e. part of the sequence is given. In standard decoding,
        labels=None."""
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)
        orig_labels = labels

        if beam_size == 1:
            decode_strategy = GreedySearch(
                sampling_temp=0.0, keep_topk=1, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False, return_hidden=True)
        else:
            decode_strategy = BeamSearch(
                beam_size=beam_size, n_best=n_best, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False)

        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        _, memory_bank = decode_strategy.initialize(memory_bank=memory_bank)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            tgt = decode_strategy.current_predictions.view(-1, 1, 1)
            if labels is not None:
                label = labels[:, step].view(-1, 1, 1)
                mask = label.eq(MASK_ID).long()
                tgt = tgt * mask + label * (1 - mask)
            tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
            dec_out, dec_attn, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank,
                                                 tgt_pad_mask=tgt_pad_mask, step=step)

            attn = dec_attn.get("std", None)

            dec_logits = self.output_layer(dec_out)  # [b, t, h] => [b, t, v]
            dec_logits = dec_logits.squeeze(1)
            log_probs = F.log_softmax(dec_logits, dim=-1)

            if self.tokenizer.output_constraint:
                output_mask = [self.tokenizer.get_output_mask(id) for id in tgt.view(-1).tolist()]
                output_mask = torch.tensor(output_mask, device=log_probs.device)
                log_probs.masked_fill_(output_mask, -10000)

            label = labels[:, step + 1] if labels is not None and step + 1 < labels.size(1) else None
            decode_strategy.advance(log_probs, attn, dec_out, label)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished:
                # Reorder states.
                memory_bank = memory_bank.index_select(0, select_indices)
                if labels is not None:
                    labels = labels.index_select(0, select_indices)
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores  # fixed to be average of token scores
        results["token_scores"] = decode_strategy.token_scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["hidden"] = decode_strategy.hidden
        results["image_bank"] = self.enc_transform(encoder_out)
        if orig_labels is not None:
            for i in range(batch_size):
                pred = results["predictions"][i][0]
                label = orig_labels[i][1:len(pred) + 1]
                mask = label.eq(MASK_ID).long()
                pred = pred[:len(label)]
                results["predictions"][i][0] = pred * mask + label * (1 - mask)

        return results["predictions"], results['scores'], results["token_scores"], results["hidden"], results["image_bank"]

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])




class GraphPredictor(nn.Module):
    """
    Predicts bond types (and optionally coordinates) from atom-level decoder hidden states,
    with optional cross-attention over image features.

    Args:
        decoder_dim (int): Hidden state dimensionality. Default 256.
        coords (bool): If True, also predict 2D atom coordinates. Default False.
    """
    
    def __init__(self, decoder_dim=256, coords=False):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.coords = coords
     
        self.image_aggregator = nn.MultiheadAttention(
            decoder_dim, 8, batch_first=True, dropout=0.1
        )
        

        self.fusion_weight = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        

        self.norm = nn.LayerNorm(decoder_dim)
        
  
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 7)  
        )
        

        if coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, 2)
            )
        self.query_proj = nn.Linear(decoder_dim, decoder_dim)

    def forward(self, hidden, memory_bank=None, indices=None):
        """
        Args:
            hidden (Tensor): Decoder hidden states of shape (B, L, D).
            memory_bank (Tensor, optional): Image feature bank of shape (B, S, D)
                                            for cross-attention. Default None.
            indices (Tensor, optional): Atom token indices into hidden, shape (B, n_atoms).
                                        If None, every 3rd token starting at position 3 is used.

        Returns:
            dict: 
                - 'edges' (Tensor): Bond logits of shape (B, 7, n_atoms, n_atoms).
                - '_atom_pairs' (Tensor): Pairwise atom features of shape (B, n_atoms, n_atoms, D).
                - 'coords' (Tensor, optional): Predicted coordinates (B, n_atoms, 2), if coords=True.
        """
        b, l, dim = hidden.size()
        
       
        if indices is None:
            
            index = [i for i in range(3, l, 3)]
            atom_features = hidden[:, index]  # (B, n_atoms, D)
        else:
            
            batch_id = torch.arange(b, device=hidden.device).unsqueeze(1).expand_as(indices).reshape(-1)
            indices_flat = indices.view(-1)
            atom_features = hidden[batch_id, indices_flat].view(b, -1, dim)
        
        q = self.query_proj(atom_features)
       
        if memory_bank is not None:
            
            enhanced_atoms, attention_weights = self.image_aggregator(
                query=q,      # (B, n_atoms, D)
                key=memory_bank,          # (B, 144, D)  
                value=memory_bank         # (B, 144, D)
            )
            
         
            
            atom_features = self.norm(
                atom_features + self.fusion_weight * enhanced_atoms
            )
        
   
        b, n_atoms, dim = atom_features.size()
        
       
        atom_pairs = torch.cat([
            atom_features.unsqueeze(2).expand(b, n_atoms, n_atoms, dim), 
            atom_features.unsqueeze(1).expand(b, n_atoms, n_atoms, dim)   
        ], dim=3)  # (B, n_atoms, n_atoms, 2*D)
        
      
        edge_feature = self.mlp[0](atom_pairs)
        edge_feature = self.mlp[1](edge_feature)
        edge_logits = self.mlp[2](edge_feature)
        # edge_logits = self.mlp(atom_pairs)  # (B, n_atoms, n_atoms, 7)
        edges = edge_logits.permute(0, 3, 1, 2)  # (B, 7, n_atoms, n_atoms)
        
     
        results = {'edges': edges}
        results["_atom_pairs"] = edge_feature
      
        if self.coords:
            results['coords'] = self.coord_mlp(atom_features)
            
        return results



def get_edge_prediction(edge_prob):
    """
    Symmetrize edge probabilities and return predicted bond types and confidence scores.

    Averages symmetric bond probabilities, with special handling for directional
    stereo bonds (types 5 and 6, i.e. wedge/dash).

    Args:
        edge_prob (list): Shape (n, n, 7) nested list of per-bond-type probabilities.

    Returns:
        tuple[list, list, list]: 
            - prediction: (n, n) argmax bond type indices.
            - score: (n, n) max probabilities.
            - edge_prob: Symmetrized probability array.
            Returns ([], [], []) if input is empty.
    """
    if not edge_prob:
        return [], [],[]
    n = len(edge_prob)
    if n == 0:
        return [], [],[]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    prediction = np.argmax(edge_prob, axis=2).tolist()
    score = np.max(edge_prob, axis=2).tolist()
    return prediction, score, edge_prob


class Decoder(nn.Module):
    """This class is a wrapper for different decoder architectures, and support multiple decoders."""

    def __init__(self, args, tokenizer):
        super(Decoder, self).__init__()
        self.args = args
        self.formats = args.formats
        self.tokenizer = tokenizer
        decoder = {}
        for format_ in args.formats:
            if format_ == 'edges':
                decoder['edges'] = GraphPredictor(args.dec_hidden_size, coords=args.continuous_coords)
            else:
                decoder[format_] = TransformerDecoderAR(args, tokenizer[format_])
        self.decoder = nn.ModuleDict(decoder)
        self.compute_confidence = args.compute_confidence
        
    def forward(self, encoder_out, hiddens, refs, mmd = False):
        """Training mode. Compute the logits with teacher forcing."""
        results = {}
        refs = to_device(refs, encoder_out.device)
        if not mmd:
            for format_ in self.formats:
                if format_ == 'edges':
                    if 'chartok_coords' in results:
                        dec_out = results['chartok_coords'][2]
                        image_memory = results['chartok_coords'][4]
                        predictions = self.decoder['edges'](dec_out, image_memory, indices=refs['atom_indices'][0])
                    else:
                        raise NotImplemented
                    targets = {'edges': refs['edges']}
                    if 'coords' in predictions:
                        targets['coords'] = refs['coords']
                    results['edges'] = (predictions, targets)
                else:
                    labels, label_lengths = refs[format_]
                    results[format_] = self.decoder[format_](encoder_out, labels, label_lengths, need_heatmap =True)
        else:
            for format_ in self.formats:
                if format_ == 'edges':
                    
                    if 'chartok_coords' in results:
                        dec_out = results['chartok_coords'][0]
                        atom_indx = results['chartok_coords'][1]
                        image_embedding = results['chartok_coords'][2]
                        # print(dec_out.shape,atom_indx.shape)
                        predictions = self.decoder['edges'](dec_out,image_embedding, indices=atom_indx)
                    else:
                        raise NotImplemented
                    
                    results['edges'] = predictions
                else:
                    results[format_] = self.decoder[format_].inference(encoder_out)
        return results

    def decode(self, encoder_out, hiddens=None, refs=None, beam_size=1, n_best=1):
        """Inference mode. Call each decoder's decode method (if required), convert the output format (e.g. token to
        sequence). Beam search is not supported yet."""
        results = {}
        predictions = []
        for format_ in self.formats:
            if format_ in ['atomtok', 'atomtok_coords', 'chartok_coords']:
                max_len = FORMAT_INFO[format_]['max_len']
                results[format_] = self.decoder[format_].decode(encoder_out, beam_size, n_best, max_length=max_len)
                outputs, scores, token_scores, *_ = results[format_]
                beam_preds = [[self.tokenizer[format_].sequence_to_smiles(x.tolist()) for x in pred]
                              for pred in outputs]
                predictions = [{format_: pred[0]} for pred in beam_preds]
                if self.compute_confidence:
                    for i in range(len(predictions)):
                        # -1: y score, -2: x score, -3: symbol score
                        indices = np.array(predictions[i][format_]['indices']) - 3
                        if format_ == 'chartok_coords':
                            atom_scores = []
                            for symbol, index in zip(predictions[i][format_]['symbols'], indices):
                                atom_score = (np.prod(token_scores[i][0][index - len(symbol) + 1:index + 1])
                                              ** (1 / len(symbol))).item()
                                atom_scores.append(atom_score)
                        else:
                            atom_scores = np.array(token_scores[i][0])[indices].tolist()
                        predictions[i][format_]['atom_scores'] = atom_scores
                        predictions[i][format_]['average_token_score'] = scores[i][0]
            if format_ == 'edges':
                if 'atomtok_coords' in results:
                    atom_format = 'atomtok_coords'
                elif 'chartok_coords' in results:
                    atom_format = 'chartok_coords'
                else:
                    raise NotImplemented
                dec_out = results[atom_format][3]  # batch x n_best x len x dim
                image_memory = results[atom_format][4]
                for i in range(len(dec_out)):
                    hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                    image_feature = image_memory[i].unsqueeze(0) 
                    indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k
                    pred = self.decoder['edges'](hidden, image_feature,indices = indices)  # k * k
                    prob = F.softmax(pred['edges'].squeeze(0).permute(1, 2, 0), dim=2).tolist()  # k * k * 7
                    edge_pred, edge_score, edge_prob = get_edge_prediction(prob)
                    predictions[i]['edges'] = edge_pred
                    # predictions[i]['_atom_pairs'] = pred["_atom_pairs"]
                    # predictions[i]["edges_prob"] = edge_prob
                    if self.compute_confidence:
                        predictions[i]['edge_scores'] = edge_score
                        predictions[i]['edge_score_product'] = np.sqrt(np.prod(edge_score)).item()
                        predictions[i]['overall_score'] = predictions[i][atom_format]['average_token_score'] * \
                                                          predictions[i]['edge_score_product']
                        predictions[i][atom_format].pop('average_token_score')
                        predictions[i].pop('edge_score_product')
        return predictions
