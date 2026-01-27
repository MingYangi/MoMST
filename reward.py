import numpy as np
import copy
from io import StringIO

import pyrosetta
from pyrosetta import rosetta
from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose
import biotite.structure.io as strucio
from tmtools import tm_align

from reward_utils import *

import requests  # ym
import json  # ym
import torch  # ym
from Bio.SeqUtils.ProtParam import ProteinAnalysis  # ym
from scipy.stats import rankdata  # ym

pyrosetta.init(options="-mute all")

_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}


def rank_normalize_scores(score_array):
    """
    使用秩转换（Rank Transformation）将一组分数客观地归一化到 [0, 1] 空间。
    最高分映射到 1.0，最低分映射到 0.0。
    该方法避免了对分数范围或主观参数（如温度）的依赖，客观地反映了奖励水平。

    参数:
        score_array (np.ndarray): 一个包含要归一化的分数的 NumPy 数组。
                                  假设分数遵循“越大越好”的原则。
    返回:
        np.ndarray: 归一化后的 [0, 1] 分数数组。
    """
    score_array = np.asarray(score_array)
    if not score_array.size:
        return np.array([])

    N = score_array.size
    if N == 1:  # 只有一个样本时，它就是当前最好的，映射为 1.0
        return np.array([1.0])

    # 1. 计算秩：使用 'min' 方法处理并列分数。最高分得到最高秩。
    # 由于原始分数已经是“越大越好”，直接计算秩即可。
    # Ranks is 1-based (1 to N)
    ranks = rankdata(score_array, method='min')

    # 2. 缩放到 [0, 1]： (秩 - 最小秩) / (最大秩 - 最小秩)
    # min_rank is 1, max_rank is N
    normalized_scores = (ranks - 1) / (N - 1)

    return normalized_scores


def esm_to_ptm(folding_result: dict, idx=0):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['ptm'].cpu().tolist()[idx] 


def esm_to_plddt(folding_result: dict, idx=0):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['mean_plddt'].cpu().tolist()[idx] * 1.0/100


def pdb_to_tm(ori_pdb_file, gen_pdb_file):
    """
    maximize tm score
    :param ori_pdb_file / gen_pdb_file: pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    seq_ori = pose_ori_pdb.sequence()
    seq_gen = pose_gen_pdb.sequence()

    ca_coor_ori = []
    for i in range(1, pose_ori_pdb.total_residue() + 1):
        if pose_ori_pdb.residue(i).has("CA"):
            ca_coord = pose_ori_pdb.residue(i).xyz("CA")
            ca_coor_ori.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_ori.append(pose_ori_pdb.sequence()[i - 1])
    ca_coor_ori = np.array(ca_coor_ori)
    # seq_ori = ''.join(seq_ori)

    ca_coor_gen = []
    for i in range(1, pose_gen_pdb.total_residue() + 1):
        if pose_gen_pdb.residue(i).has("CA"):
            ca_coord = pose_gen_pdb.residue(i).xyz("CA")
            ca_coor_gen.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_gen.append(pose_gen_pdb.sequence()[i - 1])
    ca_coor_gen = np.array(ca_coor_gen)
    # seq_gen = ''.join(seq_gen)

    tm_results = tm_align(ca_coor_ori, ca_coor_gen, seq_ori, seq_gen)
    return tm_results.tm_norm_chain1


def pdb_to_crmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    minimize rmsd, if backbone, only consider N,CA,C
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    if backbone:
        return -rosetta.core.scoring.bb_rmsd(pose_ori_pdb, pose_gen_pdb)
    else:
        return -rosetta.core.scoring.all_atom_rmsd(pose_ori_pdb, pose_gen_pdb)


def pdb_to_drmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    atom_gen = pdb_file_to_atomarray(gen_pdb_file)
    atom_ori = pdb_file_to_atomarray(ori_pdb_file)

    if backbone:
        atom_gen = get_backbone_atoms(atom_gen)
        atom_ori = get_backbone_atoms(atom_ori)

    dp = pairwise_distances(atom_gen.coord)
    dq = pairwise_distances(atom_ori.coord)

    return float(np.sqrt(((dp - dq) ** 2).mean()))


def pdb_to_lddt(ori_pdb_file, gen_pdb_file):
    """
    maximize lddt score
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    lddt = rosetta.core.scoring.lddt(pose_ori_pdb, pose_gen_pdb)
    return lddt

#计算亲水性
_HYDROPHILICS = {"ARG", "LYS", "ASP", "GLU", "ASN", "GLN", "SER", "THR", "HIS"}
def pdb_to_hydrophilic_score(gen_pdb_file, start_residue_index=None, end_residue_index=None):
    """
     maximize surface exposure
     """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)
    hydrophilic_mask = np.array([aa in _HYDROPHILICS for aa in atom_array.res_name])
    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophilic_mask)

    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophilic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array([i >= start_residue_index and i < end_residue_index for i in range(len(hydrophilic_mask))])
    hydrophilic_surf = np.logical_and(
        selection_mask * hydrophilic_mask, sasa(atom_array)
    )

    return sum(hydrophilic_surf) / sum(selection_mask*hydrophilic_mask)


def pdb_to_hydrophobic_score(gen_pdb_file, start_residue_index=None, end_residue_index=None):
    """
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface exposed
    Typically, minimize hydrophobic score
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    hydrophobic_surf = np.logical_and(
        selection_mask * hydrophobic_mask, sasa(atom_array)
    )

    return -sum(hydrophobic_surf) / sum(selection_mask * hydrophobic_mask)


def pdb_to_match_ss_score(ori_pdb_file, gen_pdb_file, start=None, end=None):
    """
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    # atom_array = strucio.load_structure(gen_pdb_file)

    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse1 = annotate_sse(subprotein)

    atom_array = pdb_file_to_atomarray(ori_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse2 = annotate_sse(subprotein)
    if len(sse1) != len(sse2):
        raise Exception("Error")
    return np.mean(sse1 == sse2), (sse1 != sse2)


def pdb_to_match_ss_score_original(ori_pdb_file, gen_pdb_file, start=None, end=None):
    """
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    # atom_array = strucio.load_structure(gen_pdb_file)

    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse1 = annotate_sse(subprotein)
    sse2 = 'a'
    return np.mean(sse1 == sse2), (sse1 != sse2)


def pdb_to_surface_expose_score(gen_pdb_file, start=None, end=None):
    """
    maximize surface exposure
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)  # 将PDB文件转换为原子数组格式，便于后续处理

    start = 0 if start is None else start  # 如果没有指定范围，则计算整个蛋白质序列的表面暴露情况
    end = len(atom_array) if end is None else end

    residue_mask = np.array([res_id in list(range(start, end)) for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))  # 使用 sasa() 函数计算每个原子的溶剂可及表面积，
    # 然后与残基掩码进行逻辑与操作，得到在指定范围内的表面暴露残基
    return sum(surface) / sum(residue_mask)


def symmetry_score(gen_pdb_file, starts, ends, all_to_all_protomer_symmetry=False):
    """
    starts: start residue index list
    ends: end residue index list

    all_to_all_protomer_symmetry: True计算全对称，False计算相邻对称
    return:
        负的距离标准差（用于最大化优化）
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    assert len(starts) == len(ends)
    centers_of_mass = []  # 计算各对称单元的重心
    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        backbone_coordinates = get_backbone_atoms(  # 提取指定区段的主链原子（N,CA,C）
            atom_array[
                np.logical_and(
                    atom_array.res_id >= start,
                    atom_array.res_id < end,
                )
            ]
        ).coord
        centers_of_mass.append(get_center_of_mass(backbone_coordinates))  # 计算该区段的重心坐标
    centers_of_mass = np.vstack(centers_of_mass)  # 堆叠所有重心坐标 [N, 3]

    # compute the standard symmetry score
    if all_to_all_protomer_symmetry:
        symmetry_score_standard = float(np.std(pairwise_distances(centers_of_mass)))
    else:
        symmetry_score_standard = float(np.std(adjacent_distances(centers_of_mass)))

    return -symmetry_score_standard

    # return (  # 计算对称性得分
    #     -float(np.std(pairwise_distances(centers_of_mass)))  # 全对称模式：计算所有重心间距离的标准差
    #     if all_to_all_protomer_symmetry
    #     else -float(np.std(adjacent_distances(centers_of_mass)))  # 相邻对称模式：计算相邻重心间距离的标准差
    # )


def pdb_to_globularity_score(gen_pdb_file, start=None, end=None):
    """
    maximize globularity score, make it as a ball
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start  # 设置计算范围的起始和结束位置，默认计算整个蛋白质序列
    end = len(atom_array) if end is None else end

    backbone = get_backbone_atoms(  # 提取指定范围内的主链原子(N, CA, C原子)坐标
        atom_array[
            np.logical_and(
                atom_array.res_id >= start,
                atom_array.res_id < end,
            )
        ]
    ).coord

    center_of_mass = get_center_of_mass(backbone)  # 计算主链原子的重心坐标
    m = backbone - center_of_mass  # 计算每个主链原子相对于重心的向量

    # computes the standard globularity score
    globularity_score_standard = float(np.std(np.linalg.norm(m, axis=-1)))  # [0, +∞) ↓

    return -globularity_score_standard  # (-∞, 0] ↑
    # return -float(np.std(np.linalg.norm(m, axis=-1)))


def pair_diversity(seq1, seq2, mask1, mask2):
    n = len(seq1)
    assert len(seq2) == n, "Sequences must be the same length"
    assert len(mask1) == n and len(mask2) == n, "Masks must be the same length as sequences"
    combined_mask = np.array(mask1) * np.array(mask2)
    effective_positions = np.sum(combined_mask)
    if effective_positions == 0:
        return 0.0

    differences = sum(1 for l in range(n) if combined_mask[l] == 1 and seq1[l] != seq2[l])
    return differences / effective_positions


def set_diversity(sequences, masks):
    m = len(sequences)
    diversity_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                diversity_matrix[i, j] = pair_diversity(sequences[i], sequences[j], masks[i], masks[j])
    overall_diversity = np.sum(diversity_matrix) / (m ** 2)
    return overall_diversity.item()

def extract_confidence_json(result_text):  # ym
    """
    使用JSON解析提取数据
    """
    try:
        # 直接解析JSON数组
        result_list = json.loads(result_text)
        if len(result_list) >= 2:
            return float(result_list[1])  # 返回第二个元素
    except json.JSONDecodeError:
        pass
    return None
def seq_to_solubility(sequence):  # ym
    """
        :param sequence: protein sequence
        :return: predicted solubility
        """
    url = "https://www.novopro.cn/plus/ppc.php"
    headers = {
        "Cookie": "_pk_id.1.24a9=0642b0864d52ba7c.1704679236.; _pk_ses.1.24a9=1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    data = {
        "sr": "sol",
        "sq": sequence
    }
    r = requests.post(url, headers=headers, data=data)
    r = r.text
    r = extract_confidence_json(r)
    return r

# def likelihood(  # ym
#     model, tokenizer,seq_len, tokenized_sample, repeat_num, device
#     ):
#     tokenized_sample = tokenized_sample
#     mask = tokenizer.mask_id
#     likelihood = torch.tensor([0.0] * repeat_num).to(device)
#     timestep = torch.tensor([0] * repeat_num)  # placeholder but not called in model
#     timestep = timestep.to(device)
#     for k in range(seq_len):
#         generated_sequence_mask = tokenized_sample.clone()
#         generated_sequence_mask[:,k] = mask
#         prediction = model(generated_sequence_mask, timestep)  # bs * seq len * dim
#         p = torch.nn.functional.softmax(prediction, dim=1)
#         likelihood += torch.log(p[ [i for i in range(repeat_num)], k, tokenized_sample[:,k]])
#     return(likelihood.detach().cpu().numpy())


def seq_to_general_fitness(  # ym
        model, tokenizer, seq_len, tokenized_sample, repeat_num, device
):
    tokenized_sample = tokenized_sample
    mask = tokenizer.mask_id
    likelihood = torch.tensor(0.0).to(device)
    timestep = torch.tensor([0])  # 修改：单条序列的时间步
    timestep = timestep.to(device)
    for k in range(seq_len):
        generated_sequence_mask = tokenized_sample.clone()
        generated_sequence_mask[k] = mask
        input_sequence = generated_sequence_mask.unsqueeze(0)  # 大多数深度学习模型期望输入是批次形式的，即使只有一条数据，也需要添加批次维度。

        prediction = model(input_sequence, timestep)  # bs(1) * seq len * dim
        p = torch.nn.functional.softmax(prediction, dim=1)
        likelihood += torch.log(p[0, k, tokenized_sample[k]])
    return likelihood.detach().cpu().numpy()


def seq_to_instability_index(sequence, target_ii=40, scale_factor=10):
    protein_analyzer = ProteinAnalysis(sequence)  # 创建蛋白质分析对象
    instability_index = protein_analyzer.instability_index()  # 计算不稳定指数
    # 使用双曲正切函数创建平滑过渡的奖励
    # 当 II < target_ii 时，奖励为正；II > target_ii 时，奖励为负
    # 输出被平滑地压缩在 (-1, 1) 区间内
    reward = -torch.tanh(torch.tensor((instability_index - target_ii) / scale_factor))
    return reward


if __name__ == "__main__":
    # ori_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/ori_pdb/T1104-D1.pdb"
    # gen_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/esm3_sm_open_v1/mcts_rollout20_depth2_posk1_sampling10_esm2_8m_esm2_8m/gen_rosettafold2/T1104-D1_idx0/models/model_00_pred.pdb"
    from biotite.database.rcsb import fetch
    import esm2

    ALL_RESIDUE_TYPES = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    RESIDUE_TYPES_WITHOUT_CYSTEINE = copy.deepcopy(ALL_RESIDUE_TYPES)
    RESIDUE_TYPES_WITHOUT_CYSTEINE.remove("C")

    template_pdb_file = fetch("6mrs", format="pdb")
    pdb_value_str = template_pdb_file.getvalue()
    template_atoms: AtomArray = pdb_file_to_atomarray(StringIO(pdb_value_str))
    sequence_length = len(sequence_from_atomarray(template_atoms))

    # random_seq = "".join([np.random.choice(RESIDUE_TYPES_WITHOUT_CYSTEINE) for _ in range(sequence_length)])

    random_seq = [np.random.choice(RESIDUE_TYPES_WITHOUT_CYSTEINE) for _ in range(sequence_length)]
    mask_indices = np.random.choice(sequence_length, size=5, replace=False)
    for idx in mask_indices:
        random_seq[idx] = "_"
    random_seq = "".join(random_seq)

    # esmfold
    # model, alphabet = esm2.esm.pretrained.esm2_t36_3B_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model.eval()  # disables dropout for deterministic results
    # # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein3", "K A <mask> I S Q"),
    # ]
    # batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # # alphabet.mask_idx: 32
    # # alphabet.padding_idx: 1. work well

    folding_model = esm2.esm.pretrained.esmfold_structure_module_only_3B().eval()
    output = folding_model.infer(random_seq)
    pdbs = folding_model.output_to_pdb(output)

    # metrics
    ptm = esm_to_ptm(output)
    plddt = esm_to_plddt(output)
    tm = pdb_to_tm(pdb_value_str, pdbs[0])
    crmsd = pdb_to_crmsd(pdb_value_str, pdbs[0])
    drmsd = pdb_to_drmsd(StringIO(pdb_value_str), StringIO(pdbs[0]))
    lddt = pdb_to_lddt(pdb_value_str, pdbs[0])
    hydrophobic = pdb_to_hydrophobic_score(StringIO(pdbs[0]))
    match_ss = pdb_to_match_ss_score(StringIO(pdbs[0]))
    surface_expose = pdb_to_surface_expose_score(StringIO(pdbs[0]))
    globularity = pdb_to_globularity_score(StringIO(pdbs[0]))
