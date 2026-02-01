import torch
import torch.nn.functional as F
import numpy as np

def debug_print_tool_plan_by_group(
    token_level_rewards: list[torch.Tensor],
    process_rewards: list[list],
    has_answer_states: list, 
    response_mask: torch.Tensor,
    index: np.ndarray,
    final_adv_tensor: torch.Tensor,
) -> None:
    """
    按 group_id (index) 把同一组样本放在一起打印，方便检查逻辑。
    Args:
        token_level_rewards: list[Tensor], len = bsz
        process_rewards:     list[list],  len = bsz
        response_mask:       Tensor,      shape = (bsz, response_length)
        index:               np.ndarray,  shape = (bsz,)
        final_adv_tensor:    Tensor,      shape = (bsz, response_length)
    """
    bsz = len(token_level_rewards)
    assert bsz == len(process_rewards) == response_mask.shape[0]  == len(index) == len(has_answer_states)
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
            print(f"  process_rewards[{i}]:     {process_rewards[i]}")
            print(f"  has_answer_states[{i}]: {has_answer_states[i]}")
            print(f"  response_mask_count[{i}]:      {get_count_from_response_mask(response_mask[i])}")
            #print(f"  response_mask[{i}]:      {response_mask[i].tolist()}")
            if final_adv_tensor != None:
              print(f"  final_adv_tensor[{i}]:   {final_adv_tensor[i].tolist()}")
            print("  ------------------------------------------")
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
  # 段号 0 用 0.0（或者 1.0，取决于你对 0 位置想要什么效果）
  factors_with_pad = torch.cat([torch.tensor([0.0], dtype=factors.dtype), factors])
  per_step_factor = factors_with_pad[segment_ids]     # 一维 [T] 的系数序列

  if debug:
    print("response_mask:", response_mask)
    print("segment_ids:  ", segment_ids)
    print("per_step_factor / scaled:", per_step_factor)
  return per_step_factor


def test_adv():
  response_mask = torch.tensor(
    [1,1,1,1,1, 0,0,0,0, 1,1,1,1,1,1, 0,0,0,0,0, 1,1,1,1],
    dtype=torch.long
  )
  factors = torch.tensor([0.2, 0.4], dtype=torch.float32)
  get_advantage_ids(response_mask, factors)
  
if __name__ == '__main__':
  test_adv()