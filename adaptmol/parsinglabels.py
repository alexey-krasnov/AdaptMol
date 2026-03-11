from tabnanny import check
from unittest import skip
from cv2 import add
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from rdkit import Chem
import math 
@dataclass
class Atom:
    """Represents an atom parsed from a MOL file atom block."""
    index: int
    x: float
    y: float
    z: float
    symbol: str
    mass_diff: int = 0
    charge: int = 0
    stereo_parity: int = 0
    hydrogen_count: int = 0
    stereo_care: int = 0
    valence: int = 0
    h0_designator: int = 0
    reaction_component_type: int = 0
    reaction_component_number: int = 0
    atom_mapping_number: int = 0
    inversion_retention_flag: int = 0
    exact_change_flag: int = 0


@dataclass
class Bond:
    """Represents a bond parsed from a MOL file bond block."""
    index: int
    atom1: int
    atom2: int
    bond_type: int
    bond_stereo: int = 0
    not_used1: int = 0
    bond_topology: int = 0
    reacting_center_status: int = 0


@dataclass
class MolData:
    """Parsed contents of a MOL file including header, atoms, bonds, and properties."""
    header: List[str]
    counts_line: str
    atoms: List[Atom]
    bonds: List[Bond]
    properties: Dict[str, Any]
    raw_content: str


def parse_mol_file(mol_path: str) -> Optional[MolData]:
    """
    Parse a MOL file into a MolData object.

    Args:
        mol_path (str): Path to the MOL file.

    Returns:
        MolData | None: Parsed mol data, or None if file is missing, empty, or malformed.
    """
    try:
        
        if not os.path.exists(mol_path):
            
            return None
            
        if os.path.getsize(mol_path) == 0:
            
            return None
        
        with open(mol_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 4:
            
            return None
        
        
        header = [line.strip() for line in lines[:3]]
        
        
        counts_line = lines[3].strip()
        if len(counts_line) < 6:
            
            return None
            
        
        try:
            parts = counts_line.split()
            atom_count = int(lines[3][:3])
            bond_count = int(lines[3][3:6])
            
        except ValueError:
            
            return None
        
        current_line = 4
        
        
        atoms = []
        for i in range(atom_count):
            if current_line >= len(lines):
                
                return None
                
            atom_line = lines[current_line]
            atom = parse_atom_line(atom_line, i + 1)
            if atom is None:
                
                return None
            atoms.append(atom)
            current_line += 1
        
        
        bonds = []
        for i in range(bond_count):
            if current_line >= len(lines):
                
                return None
                
            bond_line = lines[current_line]
            bond = parse_bond_line(bond_line, i + 1)
            if bond is None:
                
                return None
            bonds.append(bond)
            current_line += 1
        
       
        properties = {}
        while current_line < len(lines):
            line = lines[current_line].strip()
            
            
            if line.startswith('M '):
                prop_type, prop_data = parse_property_line(line)
                if prop_type:
                    
                    if prop_type in properties and prop_data is not None:
                        if isinstance(properties[prop_type], list):
                            properties[prop_type].append(prop_data)
                        else:
                            properties[prop_type].update(prop_data)
                            
                    elif prop_data is not None:
                        properties[prop_type] = prop_data
                        
                    if line == 'M  END':
                        break
            
            #
            elif line.startswith('A '):
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        atom_idx = int(parts[1])
                        
                        if current_line + 1 < len(lines):
                            alias_text = lines[current_line + 1].strip()
                            current_line += 1  
                        else:
                            alias_text = ''
                        
                        
                        if 'ALIAS' not in properties:
                            properties['ALIAS'] = {}
                        properties['ALIAS'][atom_idx] = alias_text
                        
                    except ValueError:
                        print(f"{line}")
            
            current_line += 1
        
       
        raw_content = ''.join(lines)
        
        return MolData(
            header=header,
            counts_line=counts_line,
            atoms=atoms,
            bonds=bonds,
            properties=properties,
            raw_content=raw_content
        )
        
    except Exception as e:
        
        return None


def parse_atom_line(line: str, index: int) -> Optional[Atom]:
    """
    Parse a single atom block line from a MOL file.

    Args:
        line (str): Raw atom line string.
        index (int): 1-based atom index.

    Returns:
        Atom | None: Parsed Atom object, or None on failure.
    """
    try:
       
        fields = line.split()
        
        if len(fields) < 4:  
            return None
        
        
        x = float(fields[0])
        y = float(fields[1]) 
        z = float(fields[2])
        symbol = fields[3].strip()
        
       
        atom = Atom(
            index=index,
            x=x,
            y=y,
            z=z,
            symbol=symbol
        )
        
        
        if len(fields) > 4:
            try:
                mass_diff = int(fields[4])
                atom.mass_diff = mass_diff
                
                if mass_diff != 0:
                    
                    pass
            except (ValueError, IndexError):
                atom.mass_diff = 0
        
        
        if len(fields) > 5:
            try:
                charge_code = int(fields[5])
                
                charge_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}
                atom.charge = charge_map.get(charge_code, 0)
                
                
                if charge_code == 4:
                    
                    atom.charge = 0
                    
            except (ValueError, IndexError):
                atom.charge = 0
        
        
        if len(fields) > 6:
            try:
                stereo_parity = int(fields[6])
                atom.stereo_parity = stereo_parity
                
            except (ValueError, IndexError):
                atom.stereo_parity = 0
        
        
        if len(fields) > 7:
            try:
                h_count_plus_one = int(fields[7])
                if h_count_plus_one > 0:
                    atom.hydrogen_count = h_count_plus_one - 1
                else:
                    atom.hydrogen_count = 0  
            except (ValueError, IndexError):
                atom.hydrogen_count = 0
        
        
        if len(fields) > 8:
            try:
                atom.stereo_care = int(fields[8])
               
            except (ValueError, IndexError):
                atom.stereo_care = 0
        
        
        if len(fields) > 9:
            try:
                valence = int(fields[9])
                atom.valence = valence
                
            except (ValueError, IndexError):
                atom.valence = 0
        
        
        if len(fields) > 10:
            try:
                atom.h0_designator = int(fields[10])
            except (ValueError, IndexError):
                atom.h0_designator = 0
        
        
        if len(fields) > 11:
            try:
                atom.reaction_component_type = int(fields[11])
            except (ValueError, IndexError):
                atom.reaction_component_type = 0
        
        
        if len(fields) > 12:
            try:
                atom.reaction_component_number = int(fields[12])
            except (ValueError, IndexError):
                atom.reaction_component_number = 0
        
        
        if len(fields) > 13:
            try:
                atom.atom_mapping_number = int(fields[13])
                
            except (ValueError, IndexError):
                atom.atom_mapping_number = 0
        
        
        if len(fields) > 14:
            try:
                atom.inversion_retention_flag = int(fields[14])
            except (ValueError, IndexError):
                atom.inversion_retention_flag = 0
        
        
        if len(fields) > 15:
            try:
                atom.exact_change_flag = int(fields[15])
            except (ValueError, IndexError):
                atom.exact_change_flag = 0
        
        return atom
        
    except (ValueError, IndexError) as e:
       
        return None


def parse_bond_line(line: str, index: int) -> Optional[Bond]:
    """
    Parse a single bond block line from a MOL file.

    Args:
        line (str): Raw bond line string.
        index (int): 1-based bond index.

    Returns:
        Bond | None: Parsed Bond object, or None on failure.
    """
    try:
        if len(line) < 9:  
            return None
        
        atom1 = int(line[0:3].strip())
        atom2 = int(line[3:6].strip())
        bond_type = int(line[6:9].strip())
        
        bond = Bond(
            index=index,
            atom1=atom1,
            atom2=atom2,
            bond_type=bond_type
        )
        
      
        if len(line) >= 12:
            bond.bond_stereo = int(line[9:12].strip() or 0)
        
        if len(line) >= 15:
            bond.not_used1 = int(line[12:15].strip() or 0)
        
        if len(line) >= 18:
            bond.bond_topology = int(line[15:18].strip() or 0)
        
        if len(line) >= 21:
            bond.reacting_center_status = int(line[18:21].strip() or 0)
        
        return bond
        
    except (ValueError, IndexError) as e:
        
        return None


def parse_property_line(line):
    """
    Parse an M-block property line from a MOL file.

    Supports CHG (charge), RAD (radical), ISO (isotope), and END.

    Args:
        line (str): Raw property line starting with 'M '.

    Returns:
        tuple[str | None, Any]: Property type key and parsed value dict, or (None, None) on failure.
    """
    try:
        if not line.startswith('M '):
            return None, None
        
        
        data = line[2:].strip()
        
        if len(data) < 6:
            return None, None
        
        
        prop_type = data[0:3].strip() 
        count_str = data[3:6].strip()  
        
        if not count_str.isdigit():
            return None, None
        
        count = int(count_str)
        
        if prop_type == 'CHG':  
            charges = {}
            for i in range(count):
                start_pos = 6 + i * 8  
                if start_pos + 8 <= len(data):
                    atom_idx_str = data[start_pos:start_pos+4].strip()
                    charge_str = data[start_pos+4:start_pos+8].strip()
                    
                    if atom_idx_str.isdigit() and (charge_str.isdigit() or (charge_str.startswith('-') and charge_str[1:].isdigit())):
                        atom_idx = int(atom_idx_str)
                        charge_val = int(charge_str)
                        charges[atom_idx] = charge_val
            return 'CHG', charges
        
        elif prop_type == 'RAD':  
            radicals = {}
            for i in range(count):
                start_pos = 6 + i * 8
                if start_pos + 8 <= len(data):
                    atom_idx_str = data[start_pos:start_pos+4].strip()
                    rad_str = data[start_pos+4:start_pos+8].strip()
                    
                    if atom_idx_str.isdigit() and rad_str.isdigit():
                        atom_idx = int(atom_idx_str)
                        rad_val = int(rad_str)
                        radicals[atom_idx] = rad_val
            return 'RAD', radicals
        
        elif prop_type == 'ISO':  
            isotopes = {}
            for i in range(count):
                start_pos = 6 + i * 8
                if start_pos + 8 <= len(data):
                    atom_idx_str = data[start_pos:start_pos+4].strip()
                    mass_str = data[start_pos+4:start_pos+8].strip()
                    
                    if atom_idx_str.isdigit() and mass_str.isdigit():
                        atom_idx = int(atom_idx_str)
                        mass = int(mass_str)
                        isotopes[atom_idx] = mass
            return 'ISO', isotopes
        
        elif prop_type == 'END':
            return 'END', True
        
        else:
            
            remaining_data = data[6:] if len(data) > 6 else ""
            return f'M_{prop_type}', remaining_data
    
    except (ValueError, IndexError) as e:
        
        return None, None


def atom_to_dict(atom: Atom, properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert an Atom object to a dictionary, merging in any relevant M-block properties.

    Args:
        atom (Atom): Parsed atom object.
        properties (dict, optional): Parsed M-block properties from the MOL file.

    Returns:
        dict: Atom fields plus any applicable charge, radical, isotope, and alias properties.
    """
    atom_dict = {
        'index': atom.index,
        'symbol': atom.symbol,
        'coordinates': {'x': atom.x, 'y': atom.y, 'z': atom.z},
        'mass_diff': atom.mass_diff,
        'charge': atom.charge,
        'stereo_parity': atom.stereo_parity,
        'hydrogen_count': atom.hydrogen_count,
        'stereo_care': atom.stereo_care,
        'valence': atom.valence,
        'h0_designator': atom.h0_designator,
        'reaction_component_type': atom.reaction_component_type,
        'reaction_component_number': atom.reaction_component_number,
        'atom_mapping_number': atom.atom_mapping_number,
        'inversion_retention_flag': atom.inversion_retention_flag,
        'exact_change_flag': atom.exact_change_flag
    }
    
    
    if properties:
       
        if 'CHG' in properties and atom.index in properties['CHG']:
            atom_dict['property_charge'] = properties['CHG'][atom.index]
        
        
        if 'RAD' in properties and atom.index in properties['RAD']:
            atom_dict['property_radical'] = properties['RAD'][atom.index]
            
            rad_types = {0: 'none', 1: 'singlet', 2: 'doublet', 3: 'triplet'}
            atom_dict['property_radical_type'] = rad_types.get(properties['RAD'][atom.index], 'unknown')
        
        
        if 'ISO' in properties and atom.index in properties['ISO']:
            atom_dict['property_isotope'] = properties['ISO'][atom.index]
        
        
        if 'ALIAS' in properties and atom.index in properties['ALIAS']:
            atom_dict['property_alias'] = properties['ALIAS'][atom.index]
        
        
        for prop_name, prop_value in properties.items():
            if prop_name.startswith('M_') and isinstance(prop_value, dict):
                if atom.index in prop_value:
                    atom_dict[f'property_{prop_name.lower()}'] = prop_value[atom.index]
    
    return atom_dict


def bond_to_dict(bond: Bond) -> Dict[str, Any]:
    """
    Convert a Bond object to a dictionary with human-readable type and stereo names.

    Args:
        bond (Bond): Parsed bond object.

    Returns:
        dict: Bond fields including bond_type_name and bond_stereo_name.
    """
    bond_type_names = {1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic'}
    bond_stereo_names = {0: 'not_stereo', 1: 'up', 4: 'either', 6: 'down', 3: 'cis_trans'}
    
    return {
        'index': bond.index,
        'atom1': bond.atom1,
        'atom2': bond.atom2,
        'bond_type': bond.bond_type,
        'bond_type_name': bond_type_names.get(bond.bond_type, 'unknown'),
        'bond_stereo': bond.bond_stereo,
        'bond_stereo_name': bond_stereo_names.get(bond.bond_stereo, 'unknown'),
        'bond_topology': bond.bond_topology,
        'reacting_center_status': bond.reacting_center_status
    }

def check_key(k,dic):
    if k in dic.keys():
        return True
    return False 




def get_mol(record):
    """
    Post-process atom symbol fields in a parsed mol record, applying charges,
    isotopes, aliases, mapping numbers, and bracket notation as needed.

    Args:
        record (dict): Mol record with 'atoms' and 'bonds' lists.

    Returns:
        dict: Updated record with modified atom 'symbol' fields.
    """
    atoms = record["atoms"]
    bonds = record["bonds"]
    for a in atoms:
        add_bracket = False
        has_alsis = False
        change_to_alsis = False
        symbol = a["symbol"]
        
        if check_key("property_alias", a) :
            has_alsis = True
        if has_alsis and symbol == "N":
            if not change_to_alsis:
                symbol = a["property_alias"]
                change_to_alsis =True
            add_bracket =True 
        if a["valence"] != 0 or check_key("property_radical",a):
            if has_alsis and not change_to_alsis:
                symbol = a["property_alias"]
                change_to_alsis =True
            add_bracket =True 
        if check_key("property_charge", a):
            if has_alsis and not change_to_alsis:
                if a["property_charge"] > 1:
                    symbol = a["property_alias"] + str(a["property_charge"])
                elif a["property_charge"] <-1:
                    symbol = a["property_alias"] + str( abs(a["property_charge"]))
                change_to_alsis =True                  
            elif not has_alsis:
                if a["property_charge"] ==  1:
                    symbol = symbol + "+"
                elif a["property_charge"]  == -1:
                    symbol = symbol + "-"
                if a["property_charge"] > 1:
                    symbol = symbol + "+" + str(a["property_charge"])
                if a["property_charge"] <-1:
                    symbol = symbol + "-" + str( abs(a["property_charge"]))
            else:
                if a["property_charge"] ==  1:
                    symbol = symbol 
                elif a["property_charge"]  == -1:
                    symbol = symbol 
                if a["property_charge"] > 1:
                    symbol = symbol  + str(a["property_charge"])
                if a["property_charge"] <-1:
                    symbol = symbol  + str( abs(a["property_charge"]))
            add_bracket = True
        if check_key("property_isotope",a):
            
            if has_alsis and not change_to_alsis:
                symbol = str(a["property_isotope"]) + a["property_alias"]
                change_to_alsis = True
            else :
                symbol = str(a["property_isotope"]) + symbol
            
            add_bracket =True 
        if a["atom_mapping_number"] != 0 :
            if has_alsis and not change_to_alsis:
                symbol = a["property_alias"] + ":" + str(a["atom_mapping_number"])
                change_to_alsis = True
            else :
                symbol =  symbol + ":" + str(a["atom_mapping_number"])
            add_bracket = True 
        if symbol == "R":
            if has_alsis and not change_to_alsis:
                symbol = a["property_alias"] 
                change_to_alsis = True
            else :
                symbol =  symbol 
            add_bracket = True 
        if add_bracket:
            symbol = "[" + symbol +"]"
        
        a["symbol"] = symbol
        

    return record

def process_csv_mol_data(mol_path):
    """
    Parse a MOL file and return a structured record with atoms, bonds, and metadata.

    Args:
        mol_path (str): Path to the MOL file.

    Returns:
        dict: Record with keys 'atoms', 'bonds', 'mol_data', 'status',
              'atom_count', 'bond_count', and 'mol_path'.
    """                    
    
    record = {
    
        'mol_path': mol_path,
        'mol_data': None,
        'atoms': [],
        'bonds': [],
        'status': 'failed'
    }
    
    

    
    mol_data = parse_mol_file(mol_path)
    
    if mol_data is not None:
        record['mol_data'] = mol_data
        record['atoms'] = [atom_to_dict(atom, mol_data.properties) for atom in mol_data.atoms]
        record['bonds'] = [bond_to_dict(bond) for bond in mol_data.bonds]
        record['status'] = 'success'
        record['atom_count'] = len(mol_data.atoms)
        record['bond_count'] = len(mol_data.bonds)
        record = get_mol(record)
        

               
         
            
        

    return record
    


from dataclasses import dataclass
import numpy as np

def sort_coords_and_update_bonds(coords_list, label_list, bonds_dict_list):
    """
    Sort atoms by (y, x) coordinate order and update bond atom indices accordingly.

    Args:
        coords_list (list): Atom (x, y) coordinates.
        label_list (list[str]): Atom symbol labels.
        bonds_dict_list (list[dict]): Bond records with 1-based 'atom1' and 'atom2' indices.

    Returns:
        tuple[list, list[str], list[dict]]: Sorted coordinates, labels, and updated bond dicts
                                            with 0-based atom indices.
    """
    coords_array = np.array(coords_list)
    
    
    sorted_indices = np.lexsort((coords_array[:, 0], coords_array[:, 1]))
    
   
    sorted_coords = coords_array[sorted_indices].tolist()
    sorted_labels = [label_list[i] for i in sorted_indices]

    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

    updated_bonds = []
    for bond_dict in bonds_dict_list:
        old_atom1_idx = bond_dict['atom1'] - 1
        old_atom2_idx = bond_dict['atom2'] - 1
        new_atom1 = index_mapping[old_atom1_idx]
        new_atom2 = index_mapping[old_atom2_idx] 

        new_bond_dict = bond_dict.copy()
        new_bond_dict['atom1'] = new_atom1
        new_bond_dict['atom2'] = new_atom2
        updated_bonds.append(new_bond_dict)

    return sorted_coords, sorted_labels, updated_bonds









