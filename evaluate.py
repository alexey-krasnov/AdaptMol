










import json
import argparse
import numpy as np
import multiprocessing
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs

rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--pred_field', type=str, default='SMILES')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--tanimoto', action='store_true')
    parser.add_argument('--keep_main', action='store_true')
    args = parser.parse_args()
    return args


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    """
    Canonicalize a SMILES string using RDKit.

    Args:
        smiles (str): Input SMILES string.
        ignore_chiral (bool): If True, ignore chirality during canonicalization. Default False.
        ignore_cistrans (bool): If True, remove cis/trans bond notation ('/' and '\\'). Default False.
        replace_rgroup (bool): If True, replace R-group tokens (e.g. [R1]) with wildcard atoms. Default True.

    Returns:
        tuple[str, bool]: Canonicalized SMILES and a success flag.
                          Returns ('', False) if input is invalid or empty.
    """
    if type(smiles) is not str or smiles == '' or pd.isna(smiles) or smiles == "" or str(smiles) == "nan":
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        c_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))
    except Exception as e:
        c_smiles = smiles
    try:
        
        canon_smiles = Chem.CanonSmiles(c_smiles, useChiral=(not ignore_chiral))
        success = True
    except Exception as e:
        canon_smiles = c_smiles
        success = False
    return canon_smiles, success


def convert_smiles_to_canonsmiles(
        smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=2):
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)


def _keep_main_molecule(smiles, debug=False):
    """
    Retain only the largest fragment from a SMILES string.

    Args:
        smiles (str): Input SMILES, potentially containing multiple fragments (e.g. salts).
        debug (bool): Unused, reserved for future debug logging. Default False.

    Returns:
        str: SMILES of the largest fragment by atom count, or the original string if parsing fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        pass
    return smiles


def keep_main_molecule(smiles, num_workers=2):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results


def tanimoto_similarity(smiles1, smiles2):
    """
    Compute Tanimoto similarity between two molecules using RDKit fingerprints.

    Args:
        smiles1 (str): SMILES string of the first molecule.
        smiles2 (str): SMILES string of the second molecule.

    Returns:
        float: Tanimoto similarity score in [0, 1], or 0 if either molecule fails to parse.
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto
    except Exception as e:

        return 0


def compute_tanimoto_similarities(gold_smiles, pred_smiles, num_workers=2):
    with multiprocessing.Pool(num_workers) as p:
        similarities = p.starmap(tanimoto_similarity, [(gs, ps) for gs, ps in zip(gold_smiles, pred_smiles)])
    return similarities

def convert_molfile_to_canonsmiles(
       molfile):
    smiless = []
    for i in molfile:
        try :
            mol = Chem.MolFromMolFile(i, sanitize =True)
            smiles = Chem.MolToSmiles(mol)
            smiles = smiles.replace('/', '').replace('\\', '')
        except Exception as e: 
            smiles= "C"
        smiless.append(smiles)
    return smiless
    
def molfile_to_smiles(mol_path):
    try :
        mol = Chem.MolFromMolFile(mol_path, sanitize =True)
        smiles = Chem.MolToSmiles(mol)
        smiles = smiles.replace('/', '').replace('\\', '')
        success = True
    except Exception as e: 
        mol = Chem.MolFromMolFile(mol_path, sanitize =True)
        smiles = Chem.MolToSmiles(mol)
        smiles = smiles.replace('/', '').replace('\\', '')
        success = False
    return smiles, success





class SmilesEvaluator(object):
    def __init__(self, gold_smiles, num_workers=2, tanimoto=False, mol_path = None):
        self.gold_smiles = gold_smiles
        self.num_workers = num_workers
        self.tanimoto = tanimoto
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_cistrans=True,
                                                                     num_workers=num_workers)
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                   ignore_chiral=True, ignore_cistrans=True,
                                                                   num_workers=num_workers)
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)
        self.gold_molfile_smiles = None 
        if mol_path is not None :
            self.gold_molfile_smiles = convert_molfile_to_canonsmiles(mol_path)
        

    def _replace_empty(self, smiles_list):
        """Replace empty SMILES in the gold, otherwise it will be considered correct if both pred and gold is empty."""
        return [smiles if smiles is not None and type(smiles) is str  else "<empty>"
                for smiles in smiles_list]

    def evaluate(self, pred_smiles, include_details=True):
        results = {}
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                                ignore_cistrans=True,
                                                                num_workers=self.num_workers)
        if self.tanimoto:
            canno = compute_tanimoto_similarities(self.gold_smiles, pred_smiles_cistrans)
            if self.gold_molfile_smiles is not None:
                molfile_tani = compute_tanimoto_similarities(self.gold_molfile_smiles, pred_smiles_cistrans)
                avg = (np.array(molfile_tani) + np.array(canno)) / 2
                results['tanimoto'] = np.mean(avg)
            else:
                results['tanimoto'] = np.mean(canno)

        # Ignore double bond cis/trans
        
        inchi_gt = []
        inchi_our = []
        inchi_molfile = []
        for i in range(len(self.gold_smiles_cistrans)):
            if self.gold_smiles_cistrans[i] == "":
                self.gold_smiles_cistrans[i] = pred_smiles_cistrans[i]
        for i in range(len(self.gold_smiles_cistrans)):
            try :
                mol = Chem.MolFromSmiles(self.gold_smiles_cistrans[i])
                inchi = Chem.MolToInchi(mol )
                if inchi == "":
                    inchi = "--"
                inchi_gt.append(inchi)
            except Exception as e:
                inchi_gt.append("--")
            try :
                mol1 = Chem.MolFromSmiles(pred_smiles_cistrans[i])
                inchi1 = Chem.MolToInchi(mol1)
                if inchi1 == "":
                    inchi1 = ""
                inchi_our.append(inchi1)
            except Exception as e:
                inchi_our.append("")
            if self.gold_molfile_smiles is not None:
                if self.gold_molfile_smiles[i] == "":
                    self.gold_molfile_smiles[i] = pred_smiles_cistrans[i]
                try:
                    mol2 = Chem.MolFromSmiles(self.gold_molfile_smiles[i])
                    inchi2 = Chem.MolToInchi(mol2)
                    if inchi2 == "":
                        inchi2 = "-"
                    inchi_molfile.append(inchi2)
                except Exception as e:
                    inchi_molfile.append("-")
                
        results["gt_smiles"] = self.gold_smiles_cistrans
        results["inchi"] = np.mean((np.array(inchi_gt) == np.array(inchi_our)) | (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans)))
        results["pred_smiles"] = pred_smiles_cistrans
        results['canon_smiles'] = np.mean(np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        results["molfile_smiles"] = np.mean(np.array(self.gold_smiles_cistrans) == np.array(self.gold_molfile_smiles))
        
        
        if self.gold_molfile_smiles is not None:
            results["detailmofile"] =  (np.array(self.gold_molfile_smiles) == np.array(pred_smiles_cistrans))
            molfile_matches = np.array(self.gold_molfile_smiles) == np.array(pred_smiles_cistrans)
            canon_smiles_matches = np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans)
            inchi_matches = np.array(inchi_gt) == np.array(inchi_our)
            combined_matches = molfile_matches | canon_smiles_matches # inchi_matches
            results["combined_detail"] = combined_matches
            results['combined'] = np.mean(combined_matches)

            results['molfile'] = np.mean(np.array(self.gold_molfile_smiles) == np.array(pred_smiles_cistrans))
            results["mofile_inchi"] = np.mean((np.array(inchi_molfile) == np.array(inchi_our))| (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans)))
            results["mofile_inchi_detail"] = (np.array(inchi_molfile) == np.array(inchi_our))| (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))



        else:
            canon_smiles_matches = np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans)
            inchi_matches = np.array(inchi_gt) == np.array(inchi_our)
            combined_matches = canon_smiles_matches | inchi_matches
            results['combined'] = np.mean(combined_matches)
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                              ignore_chiral=True, ignore_cistrans=True,
                                                              num_workers=self.num_workers)
        if include_details:
            results['canon_smiles_details'] = (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
            results["chiral_smiles_details"] = (np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral))
            results["inchi_details"] = (np.array(inchi_gt) == np.array(inchi_our))
            results["gt_inchi"] = np.array(inchi_gt)
            results["pred_inchi"] = np.array(inchi_our)
        # Ignore chirality (Graph exact match)
        

        results['graph'] = np.mean(np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral))

        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if '@' in g])

        results['chiral'] = np.mean(chiral[:, 0] == chiral[:, 1]) if len(chiral) > 0 else -1
        if self.gold_molfile_smiles is not None:
            results["gtmolfile"] = self.gold_molfile_smiles
        
        return results


if __name__ == "__main__":
    args = get_args()
    gold_df = pd.read_csv(args.gold_file)
    pred_df = pd.read_csv(args.pred_file)

    if len(pred_df) != len(gold_df):
        print(f"Pred ({len(pred_df)}) and Gold ({len(gold_df)}) have different lengths!")

    # Re-order pred_df to have the same order with gold_df
    image2goldidx = {image_id: idx for idx, image_id in enumerate(gold_df['image_id'])}
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    for image_id in gold_df['image_id']:
        # If image_id doesn't exist in pred_df, add an empty prediction.
        if image_id not in image2predidx:
            pred_df = pred_df.append({'image_id': image_id, args.pred_field: ""}, ignore_index=True)
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    pred_df = pred_df.reindex([image2predidx[image_id] for image_id in gold_df['image_id']])

    evaluator = SmilesEvaluator(gold_df['SMILES'], args.num_workers, args.tanimoto)
    scores = evaluator.evaluate(pred_df[args.pred_field])
    print(json.dumps(scores, indent=4))
