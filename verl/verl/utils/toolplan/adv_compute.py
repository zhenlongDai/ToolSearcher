import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

def debug_print_tool_plan_by_group(
    token_level_rewards: list[torch.Tensor],
    adv_scores: list,
    turns_tensors, 
    apiswise_adv_scores,
    search_tensors,
    has_answer_states: list, 
    response_mask: torch.Tensor,
    index: np.ndarray,
    final_adv_tensor: torch.Tensor,
) -> None:
    """
    按 group_id (index) 把同一组样本放在一起打印，方便检查逻辑。
    Args:
        token_level_rewards: list[Tensor], len = bsz
        adv_scores:     list[list],  len = bsz
        response_mask:       Tensor,      shape = (bsz, response_length)
        index:               np.ndarray,  shape = (bsz,)
        final_adv_tensor:    Tensor,      shape = (bsz, response_length)
    """
    bsz = len(token_level_rewards)
    assert bsz == len(adv_scores) == response_mask.shape[0]  == len(index) == len(has_answer_states)
    # 先按 group_id 收集样本索引
    group2idx: dict[int, list[int]] = {}
    for i, g in enumerate(index):
        g = g
        group2idx.setdefault(g, []).append(i)
    print("\n========== DEBUG TOOL_PLAN BY GROUP ==========")
    print(f"batch size: {bsz}, response_length: {response_mask.shape[1]}")
    print(f"token_level_rewards:{token_level_rewards}")
    print("----------------------------------------------")
    for g in sorted(group2idx.keys()):
        idxs = group2idx[g]
        print(f"\n>>> group_id = {g}, samples = {idxs}")
        print("----------------------------------------------")
        for i in idxs:
            print(f"[sample {i}]")
            print(f"  token_level_rewards[{i}]: {token_level_rewards[i].tolist()}")
            print(f"  turns_tensors[{i}]: {turns_tensors[i]}")
            if turns_tensors[i]!= None:
              number_turns = turns_tensors[i].shape[0]
              #print(type(turns_tensors[i]))
              print(f"number of turns is {number_turns}")
            print(f"  apiswise_adv_scores[{i}]: {apiswise_adv_scores[i]}")
            print(f"  search_tensors[{i}]: {search_tensors[i]}")
            print(f"  adv_scores[{i}]:     {adv_scores[i]}")
            print(f"  has_answer_states[{i}]: {has_answer_states[i]}")
            print(f"  response_mask_count[{i}]:      {get_count_from_response_mask(response_mask[i])}")
            #print(f"  response_mask[{i}]:      {response_mask[i].tolist()}")
            if final_adv_tensor != None:
              print(f"  final_adv_tensor[{i}]:   {final_adv_tensor[i].tolist()}")
            print("  ------------------------------------------")
            input("next")
    print("=============== END DEBUG ====================\n")

def get_count_from_response_mask(response_mask):
  # 1. 找出每一段 1 的起点
  padded = F.pad(response_mask, (1, 0), value=0)      # 左边补一个 0
  starts = (padded[1:] == 1) & (padded[:-1] == 0)     # 哪些位置是 0->1
  # 2. 给每个 1 打“段编号”：第 1 段为 1，第 2 段为 2，...
  segment_ids = torch.zeros_like(response_mask, dtype=torch.long)
  segment_ids[response_mask == 1] = torch.cumsum(starts[response_mask == 1], dim=0)
  # 3. 检查段数是否等于 factors 个数
  num_segments = int(segment_ids.max().item())
  return num_segments

def get_advantage_reward_tensor(response_mask, factors, debug=False):

  # 1. 找出每一段 1 的起点
  padded = F.pad(response_mask, (1, 0), value=0)      # 左边补一个 0
  starts = (padded[1:] == 1) & (padded[:-1] == 0)     # 哪些位置是 0->1
  # 2. 给每个 1 打“段编号”：第 1 段为 1，第 2 段为 2，...
  segment_ids = torch.zeros_like(response_mask, dtype=torch.long)
  segment_ids[response_mask == 1] = torch.cumsum(starts[response_mask == 1], dim=0)
  # 3. 检查段数是否等于 factors 个数
  num_segments = int(segment_ids.max().item())
  if factors.dim() == 0:
        factors = factors.view(1)  # one number of tensor -> (1,)
  if num_segments < factors.shape[0]:
        factors = factors[:num_segments]
  #assert num_segments == factors.shape[0], f"have {num_segments} segments \"1\"，but provide {factors.shape[0]} factors"
  # 4. 把段编号映射成系数
  # 段号0的所有位置 用0.0（取决于你对response_mask的所有0位置想要什么效果，含义为0合适）
  factors_with_pad = torch.cat([torch.tensor([0.0], dtype=factors.dtype), factors])
  per_step_factor = factors_with_pad[segment_ids]     # 一维 [T] 的系数序列

  if debug:
    print("response_mask:", response_mask)
    print("segment_ids:  ", segment_ids)
    print("per_step_factor / scaled:", per_step_factor)
  return per_step_factor




def create_turns_and_cumulative_tensors(
    length: int, 
    ids_per_turn: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据每轮的ID列表，生成一个二维的批次指示Tensor和一个一维的累积出现Tensor。

    - ID从0开始计数。
    - 如果有ID越界，将触发AssertionError。

    :param length: ID的总数，即Tensor的第二个维度长度。
    :param ids_per_turn: 一个列表，其中每个元素是对应轮次的 id 列表。
    :return: 一个元组，包含两个Tensor:
             1. (Tensor): 形状为 (num_turns, length) 的二维指示Tensor。
             2. (Tensor): 形状为 (length,) 的一维累积出现Tensor。
    """
    num_turns = len(ids_per_turn)
    if num_turns == 0:
      return None, torch.zeros(length, dtype=torch.float32)
    # 1. 初始化两个结果Tensor
    turns_indicator = torch.zeros((num_turns, length), dtype=torch.float32)
    cumulative_tracker = torch.zeros(length, dtype=torch.float32)

    # 2. 构造用于高级索引的行和列坐标
    row_indices = []
    col_indices = []
    for i, ids in enumerate(ids_per_turn):
        if ids:  # 只处理非空列表
            # 第 i 轮的所有 id，它们的行坐标都是 i
            row_indices.extend([i] * len(ids))
            # 列坐标就是 id 本身
            col_indices.extend(ids)

    # 如果所有轮次都没有提供任何ID，直接返回两个全零Tensor
    if not col_indices:
        return turns_indicator, cumulative_tracker

    # 3. 将索引列表转换为Tensor，并进行断言检查
    # col_indices_tensor 包含了所有轮次出现过的所有ID (包括重复的)
    col_indices_tensor = torch.tensor(col_indices, dtype=torch.long)
    
    # 一次性检查所有ID是否在有效范围内
    assert torch.all((col_indices_tensor >= 0) & (col_indices_tensor < length)), \
        f"Assertion Failed: Found one or more IDs outside the valid range [0, {length - 1}]."

    # 4. 高效地填充两个Tensor
    # a) 填充二维批次指示Tensor
    row_indices_tensor = torch.tensor(row_indices, dtype=torch.long)
    turns_indicator[row_indices_tensor, col_indices_tensor] = 1.0

    # b) 填充一维累积出现Tensor
    # 即使 col_indices_tensor 中有重复的ID，
    # PyTorch的索引赋值也会正确地将所有出现过的位置都设为1。
    cumulative_tracker[col_indices_tensor] = 1.0

    return turns_indicator, cumulative_tracker


def compute_groupwise_apiswise_adv_score(search_tensors, index, epsilon=1e-6):
    """
    Args:
        search_tensors: list[torch.Tensor], 每个元素 shape 为(seq_len, )
        index: list[int] or 1d np.ndarray, 和search_tensors等长，表示组别
        epsilon: 防止除0
    
    Returns:
        result_list: list[torch.Tensor]，shape 与输入batch一致
    """
    index = np.asarray(index)
    # 先对每组收集对应的tensor
    group2tensors = defaultdict(list)
    group2idxs = defaultdict(list)
    for i, gid in enumerate(index):
        group2tensors[gid].append(search_tensors[i])
        group2idxs[gid].append(i)

    # 准备结果
    result_list = [None for _ in range(len(search_tensors))]

    # 对每组进行处理
    for gid, tensors in group2tensors.items():
        # 保证同组内长度一致
        tensors = [t.cpu() for t in tensors]
        lengths = [t.size(0) for t in tensors]
        assert all(l == lengths[0] for l in lengths), f"group {gid} has unequal lengths"
        l = lengths[0]
        # 堆成矩阵: (组内样本数, seq_len)
        mat = torch.stack(tensors, dim=0)  # (bs, seq_len)
        #print(mat)
        #input()
        # 计算每一列均值、std
        mean = mat.float().mean(dim=0, keepdim=True)  # (1, seq_len)
        std = mat.float().std(dim=0, keepdim=True)    # (1, seq_len)
        # z-score归一化
        mat_zscore = (mat - mean) / (std + epsilon)
  
        # 拆成list赋值
        idxs = group2idxs[gid]
        for i, v in zip(idxs, mat_zscore):
            result_list[i] = v
    
    return result_list


def get_search_response_mask_single(response_mask, turns_tensors, selection_mask_value):
  if turns_tensors == None:
    if selection_mask_value == 0:
      search_response_mask = torch.zeros(response_mask.shape,dtype=response_mask.dtype,device=response_mask.device)
    else:
      search_response_mask = response_mask
    return search_response_mask # no any process of searching
  
  #else
  # 1. 找出每一段 1 的起点
  padded = F.pad(response_mask, (1, 0), value=0)      # 左边补一个 0
  starts = (padded[1:] == 1) & (padded[:-1] == 0)     # 哪些位置是 0->1
  # 2. 给每个 1 打“段编号”：第 1 段为 1，第 2 段为 2，...
  segment_ids = torch.zeros_like(response_mask, dtype=torch.long)
  segment_ids[response_mask == 1] = torch.cumsum(starts[response_mask == 1], dim=0)
   # 3. 检查段数是否等于 factors 个数
  num_segments = int(segment_ids.max().item())
  
  is_mask = turns_tensors.max(dim=1).values

  if is_mask.dim() == 0:
        is_mask = is_mask.view(1)  # one number of tensor -> (1,)
  if num_segments < is_mask.shape[0]:
        is_mask = is_mask[:num_segments]
  elif num_segments > is_mask.shape[0]:
        pad_val = torch.tensor(selection_mask_value,dtype=is_mask.dtype, device=is_mask.device).view(1)
        #zero = torch.zeros(1, dtype=is_mask.dtype, device=is_mask.device)
        is_mask = torch.cat([is_mask, pad_val], dim=0)
  #因为两个长度的计算方式不同，以response_mask的长度为准
  factors_with_pad = torch.cat([torch.tensor([0.0], dtype=is_mask.dtype), is_mask])
  search_response_mask = factors_with_pad[segment_ids]     # 一维 [T] 的系数序列

  return search_response_mask


def get_search_response_mask(response_mask: torch.Tensor,
                             turns_tensors: torch.Tensor,
                             selection_mask_value: float) -> torch.Tensor:
    """
    批量版本：
    response_mask: [B, T] 0/1
    turns_tensors: [B, T, ...]
    返回: [B, T]
    """

    B, T = response_mask.shape
    device = response_mask.device
    out = []
    for b in range(B):
        resp_b = response_mask[b]          # [T]
        turns_b = turns_tensors[b]        # [T, ...]
        mask_b = get_search_response_mask_single(resp_b, turns_b, selection_mask_value)  # [T]
        out.append(mask_b)
        
    search_response_mask = torch.stack(out, dim=0).to(device)  # [B, T]    
    return search_response_mask




def mask_selection_single(response_mask, turns_tensors, selection_mask_value):

  if turns_tensors == None:
    if selection_mask_value == 0:
      mask_selection = torch.zeros(response_mask.shape,dtype=response_mask.dtype,device=response_mask.device)
    else:
      mask_selection = response_mask
    return mask_selection # no any process of searching
  
  #else
  # 1. 找出每一段 1 的起点
  padded = F.pad(response_mask, (1, 0), value=0)      # 左边补一个 0
  starts = (padded[1:] == 1) & (padded[:-1] == 0)     # 哪些位置是 0->1
  # 2. 给每个 1 打“段编号”：第 1 段为 1，第 2 段为 2，...
  segment_ids = torch.zeros_like(response_mask, dtype=torch.long)
  segment_ids[response_mask == 1] = torch.cumsum(starts[response_mask == 1], dim=0)
   # 3. 检查段数是否等于 factors 个数
  num_segments = int(segment_ids.max().item())
  
  turns_mask = turns_tensors.max(dim=1).values
  is_mask = torch.ones(turns_mask.shape,dtype=turns_mask.dtype,device=turns_mask.device)
  if is_mask.dim() == 0:
        is_mask = is_mask.view(1)  # one number of tensor -> (1,)
  if num_segments < is_mask.shape[0]: # not use
        is_mask = is_mask[:num_segments]
  elif num_segments > is_mask.shape[0]:
        pad_val = torch.tensor(selection_mask_value,dtype=is_mask.dtype, device=is_mask.device).view(1)
        #zero = torch.zeros(1, dtype=is_mask.dtype, device=is_mask.device)
        is_mask = torch.cat([is_mask, pad_val], dim=0)
  #因为两个长度的计算方式不同，以response_mask的长度为准
  factors_with_pad = torch.cat([torch.tensor([0.0], dtype=is_mask.dtype), is_mask])
  mask_selection = factors_with_pad[segment_ids]     # 一维 [T] 的系数序列

  return mask_selection

def mask_selection(response_mask,
                  turns_tensors,
                  selection_mask_value) -> torch.Tensor:
    """
    批量版本：
    response_mask: [B, T] 0/1
    turns_tensors: [B, T, ...]
    返回: [B, T]
    """

    B, T = response_mask.shape
    device = response_mask.device
    out = []
    for b in range(B):
        resp_b = response_mask[b]          # [T]
        turns_b = turns_tensors[b]        # [T, ...]
        try:
          mask_b = mask_selection_single(resp_b, turns_b, selection_mask_value)  # [T]
        except Exception as e:
            # 打印详细调试信息
            print(f"[mask_selection] Error at batch index {b}")
            print(f"  response_mask[{b}].shape: {resp_b.shape}, values: {resp_b}")
            print(f"  turns_tensors[{b}].shape: {turns_b}")
            print(f"  selection_mask_value: {selection_mask_value}")
            # 如果需要，还可以打印 dtype / device
            print(f"  resp_b.dtype: {resp_b.dtype}, device: {resp_b.device}")
            print(f"  turns_b.dtype: {turns_b.dtype}, device: {turns_b.device}")
            # 再把原始异常抛出去，避免静默失败
            raise e
            
        out.append(mask_b)
        
    search_response_mask = torch.stack(out, dim=0).to(device)  # [B, T]    
    return search_response_mask



def test_compute_groupwise_apiswise_adv_score():
  # 测试样例 search_tensors 代表ID为i处的i文档是否出现过
  search_tensors = [
      torch.tensor([0,1,0,1]),
      torch.tensor([1,1,0,1]),
      torch.tensor([0,1,0,1]),
      torch.tensor([1,0,0]),    # 不同组，可以不同长度
      torch.tensor([0,0,1])
  ]
  index = [0,0,0,1,1]
  out = compute_groupwise_apiswise_adv_score(search_tensors, index)
  for o in out:
      print(o)

def test_create_batch_indicator_tensor():
  # --- 示例 1: 正常情况 ---
  length = 8
  # 轮次 0: 出现 id 1, 5
  # 轮次 1: 无
  # 轮次 2: 出现 id 0, 5, 7 (注意 5 是重复出现的)
  # 轮次 3: 出现 id 2
  ids_batch = [
      [1, 5],
      [],
      [0, 5, 7],
      [2]
  ]

  try:
      print("--- 示例 1: 正常情况 ---")
      batch_tensor, cumulative_tensor = create_turns_and_cumulative_tensors(length, ids_batch)
      
      print(f"输入: length={length}, ids_per_turn={ids_batch}")
      
      print("\n1. 二维批次指示 Tensor (shape: {}):".format(batch_tensor.shape))
      print(batch_tensor)
      
      print("\n2. 一维累积出现 Tensor (shape: {}):".format(cumulative_tensor.shape))
      print(cumulative_tensor)
      # 预期输出: ID 0, 1, 2, 5, 7 出现过 -> [1., 1., 1., 0., 0., 1., 0., 1.] ID从0开始编号

  except AssertionError as e:
      print(e)


  print("\n" + "="*50 + "\n")


  # --- 示例 2: 触发断言 ---
  length = 5
  ids_batch_invalid = [[1, 4], [0, 6]] # 这里的 '6' 是越界的

  try:
    print("--- 示例 2: 触发断言 ---")
    print(f"输入: length={length}, ids_per_turn={ids_batch_invalid}")
    create_turns_and_cumulative_tensors(length, ids_batch_invalid)
  except AssertionError as e:
    print(f"\n成功捕获到错误: {e}")


def test_adv():
  response_mask = torch.tensor(
    [1,1,1,1,1, 0,0,0,0, 1,1,1,1,1,1, 0,0,0,0,0, 1,1,1,1],
    dtype=torch.long
  )
  factors = torch.tensor([0.2, 0.4, 0.5], dtype=torch.float32)
  result = get_advantage_reward_tensor(response_mask, factors)
  print(result)

def test_get_search_response_mask():
  response_mask = torch.tensor(
    [1,1,1,1,1, 0,0,0,0, 1,1,1,1,1,1, 0,0,0,0,0, 1,1,1,1],
    dtype=torch.long
  )
  #factors = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
  length = 8
  ids_batch = [
      [1, 5],
      [],
      [0, 5, 7],
      #[2]
  ]

  batch_tensor, cumulative_tensor = create_turns_and_cumulative_tensors(length, ids_batch)
  print(f"batch_tensor:{batch_tensor}")
  print(f"response_mask:{response_mask}")
  search_response_mask = get_search_response_mask_single(response_mask, batch_tensor)
  print(search_response_mask)

if __name__ == '__main__':
  #test_adv()
  test_get_search_response_mask()
  #test_create_batch_indicator_tensor()