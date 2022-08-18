import torch
import numpy as np

from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from torch_geometric.data import Data
import joblib
import numpy as np
import math
from scipy.spatial.distance import cdist

# ===================== NODE START =====================
atomic_num_list = list(range(120))
chiral_tag_list = list(range(4))
degree_list = list(range(11))
possible_formal_charge_list = list(range(16))
possible_numH_list = list(range(9))
possible_number_radical_e_list = list(range(5))
possible_hybridization_list = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S', 'UNSPECIFIED']
possible_is_aromatic_list = [False, True]
possible_is_in_ring_list = [False, True]
implicit_valence_list = list(range(13))

def simple_atom_feature(atom):
    atomic_num = atom.GetAtomicNum()
    assert atomic_num in atomic_num_list, atomic_num

    chiral_tag = int(atom.GetChiralTag())
    assert chiral_tag in chiral_tag_list, chiral_tag
    
    degree = atom.GetTotalDegree()
    assert degree in degree_list, degree

    possible_formal_charge = atom.GetFormalCharge()
    possible_formal_charge_transformed = possible_formal_charge + 5
    assert possible_formal_charge_transformed in possible_formal_charge_list
    
    possible_numH = atom.GetTotalNumHs()
    assert possible_numH in possible_numH_list, possible_numH
    # 5
    possible_number_radical_e = atom.GetNumRadicalElectrons()
    if possible_number_radical_e > 5:
        possible_number_radical_e = 4
    assert possible_number_radical_e in possible_number_radical_e_list, possible_number_radical_e

    possible_hybridization = str(atom.GetHybridization())
    assert possible_hybridization in possible_hybridization_list, possible_hybridization
    possible_hybridization = possible_hybridization_list.index(possible_hybridization)

    possible_is_aromatic = atom.GetIsAromatic()
    assert possible_is_aromatic in possible_is_aromatic_list, possible_is_aromatic
    possible_is_aromatic = possible_is_aromatic_list.index(possible_is_aromatic)

    possible_is_in_ring = atom.IsInRing()
    assert possible_is_in_ring in possible_is_in_ring_list, possible_is_in_ring
    possible_is_in_ring = possible_is_in_ring_list.index(possible_is_in_ring)
    
    # 10
    implicit_valence = atom.GetImplicitValence()
    assert implicit_valence in implicit_valence_list, implicit_valence

    features = [
        atomic_num, chiral_tag, degree, possible_formal_charge_transformed, possible_numH,
        possible_number_radical_e, possible_hybridization, possible_is_aromatic, possible_is_in_ring,
        implicit_valence
    ]
    return features

def atom_to_feature_vector(atom):
    return simple_atom_feature(atom)

import os.path as osp
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def get_atom_feature_dims():
    feature_dims = list(map(len, [atomic_num_list, chiral_tag_list, degree_list, possible_formal_charge_list, \
                          possible_numH_list, possible_number_radical_e_list, possible_hybridization_list, \
                          possible_is_aromatic_list, possible_is_in_ring_list, implicit_valence_list])) 
    return feature_dims
# ===================== NODE END =====================

# ===================== BOND START =====================
possible_bond_type_list = list(range(32))
possible_bond_stereo_list = list(range(16))
possible_is_conjugated_list = [False, True]
possible_is_in_ring_list = [False, True]
possible_bond_dir_list = list(range(16))

def bond_to_feature_vector(bond):
    # 0
    bond_type = int(bond.GetBondType())
    assert bond_type in possible_bond_type_list

    bond_stereo = int(bond.GetStereo())
    assert bond_stereo in possible_bond_stereo_list

    is_conjugated = bond.GetIsConjugated()
    assert is_conjugated in possible_is_conjugated_list
    is_conjugated = possible_is_conjugated_list.index(is_conjugated)

    is_in_ring = bond.IsInRing()
    assert is_in_ring in possible_is_in_ring_list
    is_in_ring = possible_is_in_ring_list.index(is_in_ring)

    bond_dir = int(bond.GetBondDir())
    assert bond_dir in possible_bond_dir_list

    bond_feature = [
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        bond_dir,
    ]
    return bond_feature

def get_bond_feature_dims():
    return list(map(len, [possible_bond_type_list, possible_bond_stereo_list, possible_is_conjugated_list, possible_is_in_ring_list, possible_bond_dir_list]))
    
# ===================== BOND END =====================

def smiles2graph(mol, is_random=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    # attn
    # atom_poses = get_rel_pos(mol)
    # graph.pos = torch.tensor(atom_poses, dtype=torch.float32)
    graph = Data()
    graph.edge_index = torch.tensor(edge_index, dtype=torch.long)
    graph.edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    graph.x = torch.tensor(x, dtype=torch.long)
    graph.num_nodes = torch.tensor(len(x), dtype=torch.long)
    
    return graph

if __name__ == '__main__':
    print(len(get_atom_feature_dims()), len(get_bond_feature_dims()))
    graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    print(graph.x.shape)
    print(graph.edge_attr.shape)
