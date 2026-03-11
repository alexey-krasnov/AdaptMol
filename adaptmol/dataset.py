import os
import cv2
import time
import random
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2
import uuid
import matplotlib.pyplot as plt
from .augment import  CropWhite,SaltAndPepperNoise
from .utils import FORMAT_INFO, print_rank_0
from .tokenizer import PAD_ID
from .chemistry import  normalize_nodes
from .constants import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, COLORS
from .parsinglabels import *
import json
import traceback
from SmilesPE.pretokenizer import atomwise_tokenizer
cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


def get_transforms(input_size, augment=False, rotate=False, debug=False, pad=20,need_crop=True):
    """
    Build an Albumentations transform pipeline for image preprocessing.

    Args:
        input_size (int): Target image size (square).
        augment (bool): If True, add downscale, blur, and noise augmentations. Default False.
        rotate (bool): Unused, reserved. Default False.
        debug (bool): If True, skip normalization and tensor conversion. Default False.
        pad (int): Padding for CropWhite transform. Default 20.
        need_crop (bool): If True, apply CropWhite. Default True.

    Returns:
        A.Compose: Composed transform pipeline with keypoint support.
    """
    trans_list = []
    
    if need_crop:
        trans_list.append(CropWhite(pad=pad))
        
    if augment:
        trans_list += [
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_our_transforms( debug=False):
    """
    Build a minimal transform pipeline (grayscale, normalize, to tensor).

    Args:
        debug (bool): If True, skip all transforms. Default False.

    Returns:
        A.Compose: Composed transform pipeline with keypoint support.
    """
    trans_list = []
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))





def process_atom_tokens(token_list):
    """
    Filter and clean atom tokens from a tokenized SMILES list.

    Keeps only atom tokens and strips chirality annotations (e.g. [C@@H] -> C).

    Args:
        token_list (list[str]): Atomwise-tokenized SMILES tokens.

    Returns:
        list[str]: Cleaned atom symbol list.
    """
    UNK = '<unk>'  
    
    def is_atom_token(token):
        return token.isalpha() or token.startswith("[") or token == '*' or token == UNK
    
    def convert_stereochemistry(token):
     
       
        pattern = r'^\[(\d*)([A-Za-z\*]+)@+[Hh]?(.*)\]$'
        match = re.match(pattern, token)
        if not match:
            return token  
        number_prefix = match.group(1)  
        element = match.group(2)        
        suffix = match.group(3)         
        
        
        if element in 'Cc' and not number_prefix and not suffix:
            return 'C'
        
        elif number_prefix or suffix:
            return f'[{number_prefix}{element}{suffix}]'
        else:
            return element
    
    result = []
    for token in token_list:
        if is_atom_token(token):
           
            converted_token = convert_stereochemistry(token)
            result.append(converted_token)
    
    return result

def process_tokens(tokens):
    """
    Normalize wildcard and R-group tokens to a standard format.

    Converts [*] -> [R], * -> R, and [n*] -> [Rn].

    Args:
        tokens (list[str]): List of SMILES tokens.

    Returns:
        list[str]: Tokens with normalized R-group representations.
    """
    result = []
    for token in tokens:
       
        if token == '[*]' :
            token = '[R]'
        elif token == "*" :
            token = "R"
        else:
            token = re.sub(r'\[(\d+)\*\]', r'[R\1]', token)
        result.append(token)
    return result



def sort_by_coordinates(atoms_list, coordinates, bonds_list):
    """
    Sort atoms and bonds by (y, x) coordinate order.

    Args:
        atoms_list (list[str]): Atom symbols.
        coordinates (list): Atom (x, y) coordinates.
        bonds_list (list): Bond records as [atom1_idx, atom2_idx, bond_type].

    Returns:
        tuple: Sorted atoms, coordinates (np.ndarray), and updated bond list.
    """
    coords_array = np.array(coordinates)
    indices = np.arange(len(atoms_list))
    sorted_indices = np.lexsort((coords_array[:, 0], coords_array[:, 1]))
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    sorted_atoms = [atoms_list[i] for i in sorted_indices]
    sorted_coordinates = coords_array[sorted_indices]

    updated_bonds = []
    for bond in bonds_list:
        old_idx1, old_idx2, bond_type = bond[0], bond[1], bond[2]
        new_idx1 = old_to_new[old_idx1]
        new_idx2 = old_to_new[old_idx2]
        updated_bonds.append([new_idx1, new_idx2, bond_type])
    
    return sorted_atoms, sorted_coordinates, updated_bonds    

class TrainDataset(Dataset):
    """
    Dataset for molecular structure recognition training.

    Loads images, coordinates, atom labels, and bond graphs from CSV data,
    applies image transforms, and encodes labels into tokenized sequences.

    Args:
        args: Training arguments including data_path, input_size, formats, etc.
        df (pd.DataFrame): Data DataFrame with image paths and label columns.
        tokenizer (dict): Format-keyed tokenizer instances.
        split (str): 'train' or other split name. Default 'train'.
        dynamic_indigo (bool): If True, use dynamic Indigo rendering. Default False.
        aux (bool): If True, treat as auxiliary dataset. Default False.
        psudo_label (bool): If True, use pseudo-labeled data. Default False.
        pad (int): Padding for CropWhite. Default 20.
        need_crop (bool): If True, apply CropWhite. Default True.
    """
    def __init__(self, args, df, tokenizer, split='train', dynamic_indigo=False,aux =False, psudo_label=False,pad = 20,need_crop=True):
        super().__init__()
        self.df = df
        self.split = split
        self.args = args
        self.tokenizer = tokenizer
        self.psudo_label = psudo_label
        if 'image_path' in df.columns:
            self.file_paths = df['image_path'].values

            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df['image_path']]
            #self.file_paths = self.file_paths[:10]
        elif 'file_path' in df.columns:
            self.file_paths = df['file_path'].values

            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]
            #self.file_paths = self.file_paths[:10]
        else:
            self.file_paths = None
        if 'mol_path' in df.columns and not aux:
            self.mol_paths = df['mol_path'].values

            if not self.mol_paths[0].startswith(args.data_path):
                self.mol_paths = [os.path.join(args.data_path, path) for path in df['mol_path']]
            #self.mol_paths = self.mol_paths[:10]
        else :
            self.mol_paths = None
        
        self.smiles = df['smiles'].values if 'smiles' in df.columns else None
        if self.smiles is None:
            self.smiles = df['SMILES'].values if 'SMILES' in df.columns else None
        self.formats = args.formats
        self.labelled = (split == 'train')
        if self.labelled:
            self.labels = {}
            for format_ in self.formats:
                if format_ in ['atomtok', 'inchi']:
                    field = FORMAT_INFO[format_]['name']
                    if field in df.columns:
                        self.labels[format_] = df[field].values
        self.transform = get_transforms(args.input_size,
                                        augment=(self.labelled and args.augment),pad=pad,need_crop=need_crop)
        self.out_transform = get_our_transforms(args.input_size,
                                        augment=False)
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (dynamic_indigo and split == 'train')
        self.aux = aux
        if self.labelled and not dynamic_indigo:
            if args.coords_file == 'aux_file' and aux:
                self.aux = True
                self.coords_df = df
                self.pseudo_coords = True
            else:
                self.coords_df = self.df
                self.pseudo_coords = False
        else:
            self.coords_df = None
            self.pseudo_coords = args.pseudo_coords
        
    def __len__(self):
        return len(self.df)
    def our_image_transform(self, image, coords=[], renormalize=False):
        """Apply minimal transform (no augmentation) to image."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.out_transform(image=image, keypoints=coords)
        image = augmented['image']
        
        return image
    def image_transform(self, image, coords=[], renormalize=False):
        """Apply augmentation transform and optionally normalize coordinates."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)
        image = augmented['image']
        if len(coords) > 0:
            coords = np.array(augmented['keypoints'])
            if renormalize:
                coords = normalize_nodes(coords, flip_y=False)
            else:
                _, height, width = image.shape
                coords[:, 0] = coords[:, 0] / width
                coords[:, 1] = coords[:, 1] / height
            coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        max_retries = 10  
    
        for retry in range(max_retries):
            try:
                return self.getitem(idx)
            except Exception as e:
                error_data = {
                            'fintue' : self.psudo_label,
                            'idx': idx,
                            'traceback': traceback.format_exc(),
                            "file" : self.file_paths[idx]
                        }
                with open(os.path.join(self.args.save_path, f'error_{idx}.json'), 'w') as f:
                    json.dump(error_data, f)    
                
                idx = random.randint(0, len(self.df)-1) 
                if retry == max_retries - 1:
                    
                    raise e
    def getitem(self, idx):
        """Load and process a single sample: image, coords, labels, heatmap, and sequences."""
        ref = {}
    
    
        file_path = self.file_paths[idx]
        
        while pd.isna(file_path) or file_path == "" or str(file_path) == "nan": 
            idx = random.randint(0, len(self.df)-1)
            file_path = self.file_paths[idx]
        if self.mol_paths is not None:
            mol_path = self.mol_paths[idx]
        image = cv2.imread(file_path)
        


        if image is None:
            image = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            print(file_path, 'not found!')
        if self.coords_df is not None:
            h, w, _ = image.shape
            if self.aux or self.psudo_label:
                coords = np.array(eval(self.coords_df.loc[idx, 'node_coords']))
            else:
                coords = np.array(eval(self.coords_df.loc[idx, 'keypoints']))
            
                
            if self.pseudo_coords:
                coords = normalize_nodes(coords)
            if self.aux or self.psudo_label:
                coords[:, 0] = coords[:, 0] * w
                coords[:, 1] = coords[:, 1] * h
                image, coords = self.image_transform(image, coords)
        else:
            image = self.image_transform(image)
            coords = None
        heatmap = None
        if self.labelled:
            if self.smiles is not None:
                smiles = self.smiles[idx]
            else:
                smiles = None
            
            if not self.aux and not self.psudo_label:
                record = process_csv_mol_data(mol_path)
                atoms_list = []
                for atom in record['atoms']:
                    atoms_list.append(atom['symbol'])
                sorted_coords, sorted_labels, updated_bonds = sort_coords_and_update_bonds(coords, atoms_list, record['bonds'])
                
                h, w, _ = image.shape
                sorted_coords = np.array(sorted_coords)
                sorted_coords = sorted_coords.astype(np.float32)
                heatmap = self.generate_heatmap(sorted_coords, h,w)
                heatmap = torch.Tensor(heatmap).unsqueeze(0)
                sorted_coords[:, 0] = sorted_coords[:, 0] / w
                sorted_coords[:, 1] = sorted_coords[:, 1] / h
                
                image = self.our_image_transform(image)
            elif self.psudo_label:
                heatmap = self.generate_heatmap(coords, 1,1)
                heatmap = torch.Tensor(heatmap).unsqueeze(0)
                sorted_labels = eval(self.df.loc[idx, 'node_symbols'])
                sorted_coords = coords
                updated_bonds = eval(self.df.loc[idx, 'edges'])
            else:
                smiles = smiles.replace('/', '').replace('\\', '')
                smiles_list = atomwise_tokenizer(smiles)
                
                result_list = process_atom_tokens(smiles_list)
                result_list = process_tokens(result_list)
                edges_list  = eval(self.df.loc[idx, 'edges'])
                current_coords = coords
                
                assert len(result_list) == len(current_coords)
                    
                sorted_labels, sorted_coords, updated_bonds = sort_by_coordinates(result_list,current_coords,edges_list)
                heatmap = self.generate_heatmap(sorted_coords, 1,1)
                heatmap = torch.Tensor(heatmap).unsqueeze(0)
            if 'atomtok' in self.formats:
                max_len = FORMAT_INFO['atomtok']['max_len']
                label = self.tokenizer['atomtok'].text_to_sequence(smiles, False)
                ref['atomtok'] = torch.LongTensor(label[:max_len])
            if 'atomtok_coords' in self.formats:
                if coords is not None:
                    self._process_atomtok_coords(idx, ref, smiles, coords, mask_ratio=0)
                else:
                    self._process_atomtok_coords(idx, ref, smiles, mask_ratio=1)
            if 'chartok_coords' in self.formats:
                if coords is not None:
                    self._process_chartok_coords(idx, ref, sorted_labels, sorted_coords, mask_ratio=0, bond_list = updated_bonds)
                else:
                    self._process_chartok_coords(idx, ref, smiles, mask_ratio=1)
        if self.args.predict_coords and ('atomtok_coords' in self.formats or 'chartok_coords' in self.formats):
            smiles = self.smiles[idx]
            if 'atomtok_coords' in self.formats:
                self._process_atomtok_coords(idx, ref, smiles, mask_ratio=1)
            if 'chartok_coords' in self.formats:
                self._process_chartok_coords(idx, ref, smiles, mask_ratio=1)
        return idx, image, ref, heatmap

    def _process_atomtok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        max_len = FORMAT_INFO['atomtok_coords']['max_len']
        tokenizer = self.tokenizer['atomtok_coords']
        if smiles is None or type(smiles) is not str:
            smiles = ""
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        ref['atomtok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]
        else:
            if 'edges' in self.df.columns:
                edge_list = eval(self.df.loc[idx, 'edges'])
                n = len(indices)
                edges = torch.zeros((n, n), dtype=torch.long)
                for u, v, t in edge_list:
                    if u < n and v < n:
                        if t <= 4:
                            edges[u, v] = t
                            edges[v, u] = t
                        else:
                            edges[u, v] = t
                            edges[v, u] = 11 - t
                ref['edges'] = edges
            else:
                ref['edges'] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)

    def _process_chartok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0, bond_list = None):
        """Encode character-tokenized SMILES with coordinates and bond graph into ref dict."""
        max_len = FORMAT_INFO['chartok_coords']['max_len']
        tokenizer = self.tokenizer['chartok_coords']
        if smiles is None:
            smiles = ""
        if bond_list is not None:
            label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio, tokenized = True)
        else:
            label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        if len(label) > max_len:
            raise ValueError(f"Chartok coords sequence length {len(label)} exceeds max length {max_len}")
        ref['chartok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]
        else:
            if self.psudo_label:
                edge_list = []
                n = len(indices)
                for u in range(n):
                    for v in range(u+1, n): 
                        if bond_list[u][v] != 0:
                            edge_list.append((u, v, bond_list[u][v]))
                bond_list = edge_list
                edges = torch.zeros((n, n), dtype=torch.long)
                for u, v, t in bond_list:
                    if u < n and v < n:
                        if t <= 4:
                            edges[u, v] = t
                            edges[v, u] = t
                        else:
                            edges[u, v] = t
                            edges[v, u] = 11 - t
                
                ref['edges'] = edges
            elif 'edges' in self.df.columns:
                edge_list = bond_list
                n = len(indices)
                edges = torch.zeros((n, n), dtype=torch.long)
                for u, v, t in edge_list:
                    if u < n and v < n:
                        if t <= 4:
                            edges[u, v] = t
                            edges[v, u] = t
                        else:
                            edges[u, v] = t
                            edges[v, u] = 11 - t
                ref['edges'] = edges
            elif bond_list is not None:
                n = len(indices)
                edges = torch.zeros((n, n), dtype=torch.long)
                for bond in bond_list:
                    bond_type = bond['bond_type']
                    bond_stereo = bond['bond_stereo']
                    if bond_stereo == 1:
                        bond_type = 5
                    if bond_stereo == 6: 
                        bond_type = 6
                    if bond_type <=4 :
                        edges[bond['atom1'], bond['atom2']] = bond_type
                        edges[bond['atom2'], bond['atom1']] = bond_type
                    else:
                        edges[bond['atom1'], bond['atom2']] = bond_type
                        edges[bond['atom2'], bond['atom1']] = 11 - bond_type
                ref['edges'] = edges
                
            else:
                ref['edges'] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)


    def add_gaussian(self, input, keypoint, sigma):
        """Add a 2D Gaussian blob centered at a keypoint to a heatmap tensor."""
        tmp_size = sigma * 3

        # Top-left
        x1, y1 = int(round(keypoint[0] - tmp_size)), int(round(keypoint[1] - tmp_size))

        # Bottom right
        x2, y2 = int(round(keypoint[0] + tmp_size)) + 1, int(round(keypoint[1] + tmp_size)) + 1
        if x1 >= input.shape[0] or y1 >= input.shape[1] or x2 < 0 or y2 < 0:
            return input

        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2
        
        g = torch.tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))

        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, input.shape[0]) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, input.shape[1]) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, input.shape[0])
        img_y_min, img_y_max = max(0, y1), min(y2, input.shape[1])

        input[img_y_min:img_y_max, img_x_min:img_x_max] = torch.max(input[img_y_min:img_y_max, img_x_min:img_x_max], g[g_y_min:g_y_max, g_x_min:g_x_max])
        return input

    def generate_heatmap(self, keypoints,h,w):
        """Generate a 256x256 Gaussian heatmap from a list of keypoints."""
        sigma = 1
        keypoints = keypoints.copy()
        keypoints[:, 0] = keypoints[:, 0] * (256 / w)
        keypoints[:, 1] = keypoints[:, 1] * (256 / h)

        heatmap = torch.zeros(256, 256)
        for keypoint in keypoints:
            heatmap = self.add_gaussian(heatmap, keypoint, sigma)
            if heatmap is None:
                return heatmap
        return heatmap


class AuxTrainDataset(Dataset):

    def __init__(self, args, train_df, aux_df, tokenizer):
        super().__init__()
        self.train_dataset = TrainDataset(args, train_df, tokenizer,split='train', dynamic_indigo=False)
        self.aux_dataset = TrainDataset(args, aux_df, tokenizer,split='train', dynamic_indigo=False,aux = True)

    def __len__(self):
        return len(self.train_dataset) + len(self.aux_dataset)

    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            return self.aux_dataset[idx - len(self.train_dataset)]


def pad_images(imgs):
    """
    Pad a list of image tensors to the same spatial dimensions.

    Args:
        imgs (list[Tensor]): Image tensors of shape (C, H, W).

    Returns:
        Tensor: Stacked and zero-padded batch of shape (B, C, H, W).
    """
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def bms_collate(batch):
    ids = []
    imgs = []
    heatmap = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = [k for k in formats if
                   k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'chartok_coords', 'atom_indices']]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        heatmap.append(ex[3])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    # Time
    # if 'time' in formats:
    #     refs['time'] = [ex[2]['time'] for ex in batch]
    # Coords
    if 'coords' in formats:
        refs['coords'] = pad_sequence([ex[2]['coords'] for ex in batch], batch_first=True, padding_value=-1.)
    # Edges
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in edges_list],
            dim=0)
    if heatmap[0] is not None:
        gt_heatmap = torch.stack(heatmap)
    else :
        gt_heatmap = None
    
    return ids, pad_images(imgs), refs, gt_heatmap
