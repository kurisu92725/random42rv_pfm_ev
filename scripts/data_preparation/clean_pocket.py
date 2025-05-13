import os
import argparse
from utils.data import PDBProtein



#
# 255_EML4
# Center for 930 atoms = (19.20, -2.08, -27.90)
#
# 256
# Center for 891 atoms = (-22.28, -0.15, 16.99)
#
# 257
# Center for 1114 atoms = (-24.34, -11.30, 15.05)
#
# 258
# Center for 1192 atoms = (-9.12, 1.12, -9.70)
#
# 259
# Center for 393 atoms = (-7.95, 18.21, -12.36)
#
# 260
# Center for 458 atoms = (12.05, 11.30, -26.15)
#
# 261
# Center for 907 atoms = (15.58, 1.21, -4.26)
#
# 262
#
# Center for 1283 atoms = (-10.72, -2.99, -2.09)
#
# 263
# Center for 1172 atoms = (-8.46, -3.62, -1.79)
#
#
# 264
# Center for 1042 atoms = (-8.98, 0.95, 0.26)
#
#
# 265
# Center for 1154 atoms = (-7.26, 4.36, 3.58)
#
#
# 275
# Center for 1952 atoms = (-0.19, 1.49, 5.77)
#
#
# 276
# Center for 1282 atoms = (-11.81, 7.86, 8.56)
#
#
# 277
# Center for 952 atoms = (-15.04, 13.57, 2.37)
#

def load_protein(protein_path):
    pdb_path = os.path.join(protein_path, "3aox_updated.pdb")  # 只处理 NOX5 目录下的蛋白
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    return pdb_block


def process_protein(args):
    try:
        pdb_block = load_protein(args.source)
        protein = PDBProtein(pdb_block)

        # 直接使用给定的单个坐标进行口袋筛选
        pocket_center = (-21.72, 12.63, -8.55)# 你会提供这个坐标 (176.01,73.35,169.28)  (164.81,97.09,170.78); k18: (31.57,-0.31,-14.74),(230.96,173.46,154.92);nox5 p1: (-15.66,-1.96,-9.01) p2:(8.41,-0.20,-2.15)
        #123-075-c_476: (1.11, 7.62, 9.36);158-261-e_587:(3.15, 5.18, 10.69);158-853-e_609:(4.67, 3.29, 9.63);161_400AA: (3.68, -3.98, 0.76);310_504AA: (1.75, 10.37, 7.78)
        #337_520AA: (1.66, 10.74, 7.86)
        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand_c(pocket_center, args.radius)
        )

        pocket_fn = "3aox_%d.pdb" % args.radius
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(pocket_dest), exist_ok=True)

        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        print(f"Processed NOX5.pdb with pocket saved as {pocket_fn}")
    except Exception as e:
        print(f'Exception occurred: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./only_4irg', help='Path to NOX5 folder')
    parser.add_argument('--dest', type=str, default='Final_three_proteins_pockets', help='Path to output folder')
    parser.add_argument('--radius', type=int, default=15, help='Radius around the selected atom')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    process_protein(args)

    print(f'Done processing NOX5.pdb with radius {args.radius}.')
