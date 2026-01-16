import torch
import torch.nn.functional as F




def get_advantage_ids(response_mask, factors, debug=False):

  # 1. 找出每一段 1 的起点
  padded = F.pad(response_mask, (1, 0), value=0)      # 左边补一个 0
  starts = (padded[1:] == 1) & (padded[:-1] == 0)     # 哪些位置是 0->1
  # 2. 给每个 1 打“段编号”：第 1 段为 1，第 2 段为 2，...
  segment_ids = torch.zeros_like(response_mask, dtype=torch.long)
  segment_ids[response_mask == 1] = torch.cumsum(starts[response_mask == 1], dim=0)
  # 3. 检查段数是否等于 factors 个数
  num_segments = int(segment_ids.max().item())
  assert num_segments == factors.shape[0], f"have {num_segments} segments \"1\"，but only provide {factors.shape[0]} factors"
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
  get_advantage_idsJ(response_mask, factors)
  
if __name__ == '__main__':
  test_adv()