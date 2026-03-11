import copy
import traceback
import numpy as np
import multiprocessing
import itertools

import rdkit
import rdkit.Chem as Chem

rdkit.RDLogger.DisableLog('rdApp.*')

from SmilesPE.pretokenizer import atomwise_tokenizer

from .constants import RGROUP_SYMBOLS, ABBREVIATIONS, VALENCES, FORMULA_REGEX




def normalize_nodes(nodes, flip_y=True):
    """
    Normalize node coordinates to [0, 1] range.

    Args:
        nodes (np.ndarray): Shape (N, 2) array of (x, y) coordinates.
        flip_y (bool): If True, flip y-axis. Default True.

    Returns:
        np.ndarray: Normalized coordinates of shape (N, 2).
    """
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def _verify_chirality(mol, coords, symbols, edges, debug=False):
    """
    Assign chirality and stereochemistry to a molecule using 2D coordinates and bond directions.

    Args:
        mol (Chem.RWMol): Editable molecule object.
        coords (list): 2D coordinates for each atom.
        symbols (list[str]): Atom symbols.
        edges (np.ndarray): Adjacency matrix with bond type encoding.
        debug (bool): If True, re-raise exceptions. Default False.

    Returns:
        Chem.Mol: Molecule with chirality assigned.
    """
    try:
        n = mol.GetNumAtoms()
        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        Chem.SanitizeMol(mol_tmp)

        chiral_centers = Chem.FindMolChiralCenters(
            mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers]  # List[Tuple[int, any]] -> List[int]

        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        # Magic, inferring chirality from coordinates and BondDir. DO NOT CHANGE.
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)

        # Second loop to reset any wedge/dash bond to be starting from the chiral center)
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for non-carbon atom
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    # if ''.join(tokens) != formula:
    #     tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # skip unreasonable number of atoms
        if num > 100000:
            i += 1; continue
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            if next_num > 100000:
                i += 1; continue
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded


def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """

    new = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in new or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''


def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula_list`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `num_trials`: number of trials
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1

    def dfs(smiles, bonds_left, cur_idx, add_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """
        num_trials = 1
        # end of formula: return result
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  # atom to be added to current atom

        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:
                # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # put smiles of "atom" in parentheses and append to smiles; go to next atom to add to current atom
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                # "atom" added uses up remaining bonds of current atom
                # get smiles of "atom" and bonds left on it
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success

        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, [1]):  # try all possible valences of atom added
            add_str = _expand_abbreviation(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False

    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)


def get_smiles_from_symbol(symbol, mol, atom, bonds,abb = None):
    """
    Convert symbol (abbrev. or condensed formula) to smiles
    If condensed formula, determine parsing direction and num. bonds on each side using coordinates
    """
    if abb is not None:
        if symbol in abb:
            return None
    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles
    
    
    if len(symbol) > 20:
        return None

    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))
    formula_list = _expand_carbon(_parse_formula(symbol))
    
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    
    if success:
        return smiles
    return None


def _replace_functional_group(smiles):
    """
    Replace R-groups and unknown tokens in SMILES with wildcard placeholders.

    Returns:
        tuple[str, dict]: Processed SMILES and isotope-to-symbol mapping for unknowns.
    """
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    # For unknown tokens (i.e. rdkit cannot parse), replace them with [{isotope}*], where isotope is an identifier.
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  # isotope : symbol
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            if token[1:-1] in ABBREVIATIONS or Chem.AtomFromSmiles(token) is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings


def convert_smiles_to_mol(smiles):
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol


BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def _expand_functional_group(mol, mappings, debug=False,abb=None):
    """
    Expand abbreviation/condensed formula placeholders in a molecule into full substructures.

    Args:
        mol (Chem.Mol): Input molecule with placeholder atoms.
        mappings (dict): Isotope-to-symbol mapping from _replace_functional_group.
        debug (bool): If True, re-raise exceptions. Default False.
        abb (dict, optional): Custom abbreviation set. Default None.

    Returns:
        tuple[str, Chem.Mol]: Canonical SMILES and expanded molecule.
    """
    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0

    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        num_atoms = mol_w.GetNumAtoms()
        for i, atom in enumerate(mol_w.GetAtoms()):  # reset radical electrons
            atom.SetNumRadicalElectrons(0)

        atoms_to_remove = []
        for i in range(num_atoms):
            atom = mol_w.GetAtomWithIdx(i)
            if atom.GetSymbol() == '*':
                symbol = Chem.GetAtomAlias(atom)
                isotope = atom.GetIsotope()
                if isotope > 0 and isotope in mappings:
                    symbol = mappings[isotope]
                if not (isinstance(symbol, str) and len(symbol) > 0):
                    continue
                # rgroups do not need to be expanded
                if symbol in RGROUP_SYMBOLS:
                    continue

                bonds = atom.GetBonds()
                sub_smiles = get_smiles_from_symbol(symbol, mol_w, atom, bonds,abb=abb)
                
                # create mol object for abbreviation/condensed formula from its SMILES
                mol_r = convert_smiles_to_mol(sub_smiles)

                if mol_r is None:
                    # atom.SetAtomicNum(6)
                    atom.SetIsotope(0)
                    continue

                # remove bonds connected to abbreviation/condensed formula
                adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds]
                for adjacent_idx in adjacent_indices:
                    mol_w.RemoveBond(i, adjacent_idx)

                adjacent_atoms = [mol_w.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
                for adjacent_atom, bond in zip(adjacent_atoms, bonds):
                    adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))

                # get indices of atoms of main body that connect to substituent
                bonding_atoms_w = adjacent_indices
                # assume indices are concated after combine mol_w and mol_r
                bonding_atoms_r = [mol_w.GetNumAtoms()]
                for atm in mol_r.GetAtoms():
                    if atm.GetNumRadicalElectrons() and atm.GetIdx() > 0:
                        bonding_atoms_r.append(mol_w.GetNumAtoms() + atm.GetIdx())

                # combine main body and substituent into a single molecule object
                combo = Chem.CombineMols(mol_w, mol_r)

                # connect substituent to main body with bonds
                mol_w = Chem.RWMol(combo)
                # if len(bonding_atoms_r) == 1:  # substituent uses one atom to bond to main body
                for atm in bonding_atoms_w:
                    bond_order = mol_w.GetAtomWithIdx(atm).GetNumRadicalElectrons()
                    mol_w.AddBond(atm, bonding_atoms_r[0], order=BOND_TYPES[bond_order])

                # reset radical electrons
                for atm in bonding_atoms_w:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                for atm in bonding_atoms_r:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                atoms_to_remove.append(i)

        # Remove atom in the end, otherwise the id will change
        # Reverse the order and remove atoms with larger id first
        atoms_to_remove.sort(reverse=True)
        for i in atoms_to_remove:
            mol_w.RemoveAtom(i)
        # try:
        #     Chem.AssignChiralTypesFromBondDirs(mol_w)
        #     Chem.AssignStereochemistry(mol_w, force=True)
        # except:
        #     pass
        smiles = Chem.MolToSmiles(mol_w)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol)
    return smiles, mol


    
def resolve_abbreviations_simplified(molecule, abbreviations_smiles_mapping):
    """
    Iteratively expand all abbreviation atoms in a molecule using a SMILES mapping.

    Args:
        molecule (Chem.Mol): Input molecule with aliased abbreviation atoms.
        abbreviations_smiles_mapping (dict): Mapping from abbreviation to {'smiles': str}.

    Returns:
        Chem.Mol: Molecule with abbreviations expanded.
    """
    
    current_molecule = molecule
    max_iterations = 10 
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        abbreviations_found = []
        
        
        for atom_idx, atom in enumerate(current_molecule.GetAtoms()):
            alias = Chem.GetAtomAlias(atom)
            if alias and alias in abbreviations_smiles_mapping:
                abbreviations_found.append({
                    'atom_idx': atom_idx,
                    'alias': alias,
                    'smiles': abbreviations_smiles_mapping[alias]["smiles"]
                })
        
        if not abbreviations_found:
            
            break
            
    
        abbr_info = abbreviations_found[0]
        
        result = process_single_abbreviation(
            current_molecule, 
            abbr_info['atom_idx'], 
            abbr_info['alias'], 
            abbr_info['smiles']
        )
        
        if result is None:
            
            break
            
        current_molecule = result
       
    
    return current_molecule


def process_single_abbreviation(molecule, atom_idx, alias, smiles):
    """
    Replace a single abbreviation atom with its corresponding substructure.

    Args:
        molecule (Chem.Mol): Input molecule.
        atom_idx (int): Index of the abbreviation atom to replace.
        alias (str): Expected alias of the atom.
        smiles (str): SMILES of the abbreviation's substructure.

    Returns:
        Chem.Mol | None: Updated molecule, or None on failure.
    """
    from rdkit import Chem
    
   
    
   
    if atom_idx >= molecule.GetNumAtoms() or atom_idx < 0:
        return None
    
    abbreviation_mol = Chem.MolFromSmiles(smiles)
    if abbreviation_mol is None:
        return None
    
    
    target_atom = molecule.GetAtomWithIdx(atom_idx)
    
    
    current_alias = Chem.GetAtomAlias(target_atom)
    if current_alias != alias:
    
        return None
    
    
    neighbor_bonds = []
    for bond in target_atom.GetBonds():
        other_atom_idx = bond.GetOtherAtomIdx(atom_idx)
        neighbor_bonds.append({
            'neighbor_idx': other_atom_idx,
            'bond_type': bond.GetBondType(),
            'bond_dir': bond.GetBondDir(),  
            'begin_atom_idx': bond.GetBeginAtomIdx(),  
            'end_atom_idx': bond.GetEndAtomIdx()  
        })

    processed_abbr = process_abbreviation_molecule(abbreviation_mol)
    if processed_abbr is None:
        return None
        
    clean_abbr_mol, connection_points = processed_abbr
    

    if len(neighbor_bonds) != len(connection_points):
        return None
    
   
    return build_new_molecule(molecule, atom_idx, neighbor_bonds, clean_abbr_mol, connection_points)


def process_abbreviation_molecule(abbreviation_mol):
    """
    Extract the core structure and connection points from an abbreviation molecule.

    Removes wildcard (*) atoms and identifies their neighbors as connection points.

    Args:
        abbreviation_mol (Chem.Mol): Molecule with * atoms marking attachment sites.

    Returns:
        tuple[Chem.Mol, list[dict]] | None: Core molecule and connection point info,
                                            or None on failure.
    """
    from rdkit import Chem
    
    
    star_atoms = []
    connection_data = []
    
    for atom_idx, atom in enumerate(abbreviation_mol.GetAtoms()):
        if atom.GetSymbol() == "*":
            star_atoms.append(atom_idx)
            
            
            bonds = list(atom.GetBonds())
            if len(bonds) != 1:
                
                return None
            
            bond = bonds[0]
            neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
            
            connection_data.append({
                'star_idx': atom_idx,
                'neighbor_idx': neighbor_idx,
                'bond_type': bond.GetBondType()
            })
    
    if not star_atoms:
       
        return None
    
    
    index_map = {}
    new_idx = 0
    for old_idx in range(abbreviation_mol.GetNumAtoms()):
        if old_idx not in star_atoms:
            index_map[old_idx] = new_idx
            new_idx += 1
    
    
    new_mol = Chem.EditableMol(Chem.Mol())
    
   
    for atom_idx, atom in enumerate(abbreviation_mol.GetAtoms()):
        if atom_idx not in star_atoms:
            new_mol.AddAtom(atom)
    
    
    for bond in abbreviation_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        
        if begin_idx in star_atoms or end_idx in star_atoms:
            continue
            
        
        new_begin = index_map[begin_idx]
        new_end = index_map[end_idx]
        new_mol.AddBond(new_begin, new_end, bond.GetBondType())
    
    try:
        clean_mol = new_mol.GetMol()
        if clean_mol is None:
          
            return None
        Chem.SanitizeMol(clean_mol)
    except Exception as e:
       
        return None
    

    connection_points = []
    for conn_data in connection_data:
        neighbor_idx = conn_data['neighbor_idx']
        if neighbor_idx in index_map:
            connection_points.append({
                'atom_idx': index_map[neighbor_idx],
                'bond_type': conn_data['bond_type']
            })
    
    return clean_mol, connection_points


def build_new_molecule(original_mol, target_atom_idx, neighbor_bonds, abbr_mol, connection_points):
    """
    Replace a target atom with an abbreviation substructure, preserving chirality.

    Args:
        original_mol (Chem.Mol): The original molecule.
        target_atom_idx (int): Index of atom to be replaced.
        neighbor_bonds (list[dict]): Bond info for each neighbor of the target atom.
        abbr_mol (Chem.Mol): Core structure of the abbreviation.
        connection_points (list[dict]): Attachment atoms in abbr_mol.

    Returns:
        Chem.Mol | None: New molecule with abbreviation expanded, or None on failure.
    """
    from rdkit import Chem
    
    try:
        
        atom_offset = original_mol.GetNumAtoms()
        combined_mol = Chem.CombineMols(original_mol, abbr_mol)
        
        
        rw_mol = Chem.RWMol(combined_mol)
        
       
        chiral_info = []
        for i, neighbor_bond in enumerate(neighbor_bonds):
            if i < len(connection_points):
                if neighbor_bond.get('bond_dir', Chem.BondDir.NONE) in [Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH]:
                    chiral_info.append({
                        'neighbor_idx': neighbor_bond['neighbor_idx'],
                        'connection_idx': connection_points[i]['atom_idx'] + atom_offset,
                        'bond_dir': neighbor_bond['bond_dir'],
                        'from_target': neighbor_bond['begin_atom_idx'] == target_atom_idx
                    })
        
       
        for i, neighbor_bond in enumerate(neighbor_bonds):
            if i < len(connection_points):
                neighbor_idx = neighbor_bond['neighbor_idx']
                connection_atom_idx = connection_points[i]['atom_idx'] + atom_offset
                bond_type = neighbor_bond['bond_type']
                
               
                if neighbor_bond['begin_atom_idx'] == target_atom_idx:
                   
                    rw_mol.AddBond(connection_atom_idx, neighbor_idx, bond_type)
                else:
                    
                    rw_mol.AddBond(neighbor_idx, connection_atom_idx, bond_type)
        
        
        rw_mol.RemoveAtom(target_atom_idx)
        
        
        mol = rw_mol.GetMol()
        
        
        Chem.SanitizeMol(mol)
        
        
        rw_mol = Chem.RWMol(mol)
        
        
        for bond in rw_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)
        
        
        for info in chiral_info:
           
            neighbor_idx = info['neighbor_idx']
            connection_idx = info['connection_idx']
            
            if neighbor_idx > target_atom_idx:
                neighbor_idx -= 1
            if connection_idx > target_atom_idx:
                connection_idx -= 1
            
           
            bond = rw_mol.GetBondBetweenAtoms(neighbor_idx, connection_idx)
            if bond:
                
                rw_mol.RemoveBond(neighbor_idx, connection_idx)
                
                if info['from_target']:
                    
                    rw_mol.AddBond(connection_idx, neighbor_idx, Chem.BondType.SINGLE)
                    bond = rw_mol.GetBondBetweenAtoms(connection_idx, neighbor_idx)
                else:
                    
                    rw_mol.AddBond(neighbor_idx, connection_idx, Chem.BondType.SINGLE)
                    bond = rw_mol.GetBondBetweenAtoms(neighbor_idx, connection_idx)
                
                bond.SetBondDir(info['bond_dir'])
        
        
        final_mol = rw_mol.GetMol()
        
       
        Chem.SanitizeMol(final_mol)
        Chem.AssignChiralTypesFromBondDirs(final_mol)
        Chem.AssignStereochemistry(final_mol, force=True)
        
        
        for atom in final_mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        
        return final_mol
        
    except Exception as e:
        
        return None








def convert_graph_to_smiles(coords, symbols, edges, images=None, num_workers=1,abb =None):
    
    if images is None:
        abb_repeated = [abb] * len(coords)
        args_zip = zip(coords, symbols, edges,abb_repeated)
    else:
        args_zip = zip(coords, symbols, edges, images)

    if num_workers <= 1:
        
        results = itertools.starmap(_convert_graph_to_smiles_with_chirality_fix, args_zip)
        results = list(results)
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_smiles_with_chirality_fix, args_zip, chunksize=128)

    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, debug=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    mol = None
    pred_molblock = ''
    try:
        pred_smiles = smiles
        pred_smiles, mappings = _replace_functional_group(pred_smiles)
        if coords is not None and symbols is not None and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        # pred_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if molblock:
            pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_smiles = smiles
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success


def postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        if coords is not None and symbols is not None and edges is not None:
            results = p.starmap(_postprocess_smiles, zip(smiles, coords, symbols, edges), chunksize=128)
        else:
            results = p.map(_postprocess_smiles, smiles, chunksize=128)
    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _keep_main_molecule(smiles, debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        if debug:
            print(traceback.format_exc())
    return smiles


def keep_main_molecule(smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results

def _convert_graph_to_smiles_with_chirality_fix(coords, symbols, edges, abb=None, image=None, debug=False):
    """
    Version that fixes chirality after functional group expansion by preserving wedge/dash information
    """
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
   
    for i in range(n):
        symbol = symbols[i]
        
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if len(symbol) > 0 and (symbol in RGROUP_SYMBOLS or symbol[0] == 'R' or symbol[0] == '*'):
            atom = Chem.Atom("*")
            if (symbol[0] == 'R' or symbol[0] == '*') and symbol[1:].isdigit():
                try:
                    atom.SetIsotope(int(symbol[1:]))
                except:
                    atom.SetIsotope(0)
            Chem.SetAtomAlias(atom, symbol)
            
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        elif symbols[i] == "[(]" or symbols[i] == "[)]":
            atom = Chem.Atom("C")
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        else:
            try:
                if symbols[i] == "H" or symbols[i] == "h":
                    atom = Chem.AtomFromSmiles("[H]")
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                elif "." in symbols[i]:
                    atom = Chem.Atom("*")
                else: 
                    atom = Chem.AtomFromSmiles(symbols[i])
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                
            except:
                try:
                    if symbols[i][0] != '[':
                        new_symbol = "[" + symbols[i] + "]"
                        atom = Chem.AtomFromSmiles(new_symbol)
                        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                    else:
                        atom = Chem.Atom("*")
                        Chem.SetAtomAlias(atom, symbol)
                except:
                    atom = Chem.Atom("*")
                    Chem.SetAtomAlias(atom, symbol)
        
        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    pred_smiles = '<invalid>'

    try:
        # TODO: move to an util function
        if image is not None:
            height, width, _ = image.shape
            ratio = width / height
            coords = [[x * ratio * 10, y * 10] for x, y in coords]
        
        
        chirality_bonds = _store_chirality_bonds(mol, coords)
        
        
        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        
        
        pred_molblock = Chem.MolToMolBlock(mol)
        
        
        pred_smiles, mol = _expand_functional_group(mol, {}, debug, abb=abb)
        
        
        mol = resolve_abbreviations_simplified(mol, abb)
        
       
        mol = _transfer_chirality_to_expanded_mol(mol, chirality_bonds, coords, debug)
        
        pred_smiles = Chem.MolToSmiles(mol) 
        success = True
        
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_molblock = ''
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success


def _store_chirality_bonds(mol, coords):
    """
    Store information about wedge/dash bonds before expansion
    """
    chirality_info = []
    
    for bond in mol.GetBonds():
        if bond.GetBondDir() in [Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH]:
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            begin_atom = mol.GetAtomWithIdx(begin_idx)
            end_atom = mol.GetAtomWithIdx(end_idx)
            
            # Store detailed information
            chirality_info.append({
                'begin_idx': begin_idx,
                'end_idx': end_idx,
                'begin_coord': coords[begin_idx],
                'end_coord': coords[end_idx],
                'bond_dir': bond.GetBondDir(),
                'begin_symbol': begin_atom.GetSymbol(),
                'end_symbol': end_atom.GetSymbol(),
                'begin_alias': Chem.GetAtomAlias(begin_atom),
                'end_alias': Chem.GetAtomAlias(end_atom),
                'begin_neighbors': _get_neighbor_info(mol, begin_idx, coords),
                'end_neighbors': _get_neighbor_info(mol, end_idx, coords)
            })
    
    return chirality_info


def _get_neighbor_info(mol, atom_idx, coords):
    """
    Get information about an atom's neighbors
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = []
    
    for bond in atom.GetBonds():
        neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
        neighbor_atom = mol.GetAtomWithIdx(neighbor_idx)
        neighbors.append({
            'idx': neighbor_idx,
            'symbol': neighbor_atom.GetSymbol(),
            'alias': Chem.GetAtomAlias(neighbor_atom),
            'coord': coords[neighbor_idx],
            'bond_type': bond.GetBondType()
        })
    
    return neighbors


def _transfer_chirality_to_expanded_mol(mol, chirality_bonds, original_coords, debug=False):
    """
    Transfer chirality information to the expanded molecule
    """
    try:
        # Create a conformer with estimated coordinates for the expanded molecule
        n_atoms = mol.GetNumAtoms()
        conf = Chem.Conformer(n_atoms)
        conf.Set3D(False)
        
        # Assign coordinates to expanded molecule atoms
        coords_assigned = {}
        
        # First pass: assign coordinates to atoms that clearly correspond to original atoms
        for i in range(min(n_atoms, len(original_coords))):
            conf.SetAtomPosition(i, (original_coords[i][0], 1 - original_coords[i][1], 0))
            coords_assigned[i] = original_coords[i]
        
        # Second pass: assign coordinates to new atoms based on their connections
        for i in range(len(original_coords), n_atoms):
            if i not in coords_assigned:
                # Find connected atoms that have coordinates
                atom = mol.GetAtomWithIdx(i)
                coord_found = False
                
                for bond in atom.GetBonds():
                    neighbor_idx = bond.GetOtherAtomIdx(i)
                    if neighbor_idx in coords_assigned:
                        # Use neighbor's coordinate (this is the key insight)
                        coord = coords_assigned[neighbor_idx]
                        conf.SetAtomPosition(i, (coord[0], 1 - coord[1], 0))
                        coords_assigned[i] = coord
                        coord_found = True
                        break
                
                if not coord_found:
                    # Fallback to origin
                    conf.SetAtomPosition(i, (0, 0, 0))
                    coords_assigned[i] = [0, 0]
        
        # Add conformer to molecule
        if mol.GetNumConformers() > 0:
            mol.RemoveAllConformers()
        mol.AddConformer(conf)
        
        # Now try to restore chirality bonds
        _restore_chirality_bonds(mol, chirality_bonds, coords_assigned, debug)
        
        # Clear all bond directions first
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)
        
        # Re-run chirality detection with coordinates
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)
        
        # Reset non-carbon chirality
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        
    except Exception as e:
        if debug:
            print(f"Error in chirality transfer: {e}")
            import traceback
            print(traceback.format_exc())
    
    return mol


def _restore_chirality_bonds(mol, chirality_bonds, coords_assigned, debug=False):
    """
    Restore wedge/dash bonds in the expanded molecule
    """
    for chiral_info in chirality_bonds:
        # Try to find corresponding atoms in expanded molecule
        begin_candidates = _find_atom_candidates(
            mol, chiral_info['begin_symbol'], chiral_info['begin_alias'], 
            chiral_info['begin_coord'], chiral_info['begin_neighbors'], coords_assigned
        )
        
        end_candidates = _find_atom_candidates(
            mol, chiral_info['end_symbol'], chiral_info['end_alias'],
            chiral_info['end_coord'], chiral_info['end_neighbors'], coords_assigned
        )
        
        if debug:
            print(f"Begin candidates: {begin_candidates}, End candidates: {end_candidates}")
        
        # Find the best match
        for begin_idx in begin_candidates:
            for end_idx in end_candidates:
                bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
                if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                    # Set the chirality bond direction
                    bond.SetBondDir(chiral_info['bond_dir'])
                    if debug:
                        print(f"Restored chirality bond: {begin_idx}-{end_idx} with direction {chiral_info['bond_dir']}")
                    break


def _find_atom_candidates(mol, target_symbol, target_alias, target_coord, target_neighbors, coords_assigned):
    """
    Find candidate atoms in expanded molecule that could correspond to original atom
    """
    candidates = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        # Match by symbol first
        if atom.GetSymbol() != target_symbol:
            continue
        
        # If original atom had alias, skip (it was a functional group that got expanded)
        if target_alias:
            continue
        
        # Check coordinate proximity
        if i in coords_assigned:
            atom_coord = coords_assigned[i]
            if _coords_close(atom_coord, target_coord, tolerance=0.5):
                candidates.append(i)
        else:
            # If no coordinate assigned, it's less likely but still possible
            candidates.append(i)
    
    # If no exact matches, be more lenient
    if not candidates:
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() == target_symbol:
                candidates.append(i)
    
    return candidates


def _coords_close(coord1, coord2, tolerance=0.1):
    """
    Check if two coordinates are close
    """
    return abs(coord1[0] - coord2[0]) < tolerance and abs(coord1[1] - coord2[1]) < tolerance