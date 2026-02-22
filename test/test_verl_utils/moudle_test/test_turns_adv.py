import torch

def compute_adv_scores(turns_tensors, apiswise_adv_scores):
    """
    对每个批次，对每轮，如果该轮API向量全为0，则取apiswise_adv_score最小值，否则取乘积最大值
    Args:
        turns_tensors: list[Tensor], 每个shape=(turns, apis_nums)
        apiswise_adv_scores: list[Tensor], 每个shape=(apis_nums,)
    Returns:
        adv_scores: list[Tensor], 每个shape=(turns,)
    """
    adv_scores = []
    for turns_tensor, apiswise_adv_score in zip(turns_tensors, apiswise_adv_scores):
        if turns_tensor is not None:
            mul = turns_tensor * apiswise_adv_score
            row_is_zero = (turns_tensor == 0).all(dim=1)
            row_max = mul.max(dim=1).values
            min_val = apiswise_adv_score.min()
            adv_score = torch.where(row_is_zero, min_val, row_max)
        else:
            adv_score = None
        adv_scores.append(adv_score)
    return adv_scores

# -------------------------------
# 单元测试
def test_compute_adv_scores():
    # Case 1: 通常行
    turns_tensors = [
        torch.tensor([[0,0,0],[0,1,0],[0,0,2]], dtype=torch.float32)
    ]
    apiswise_adv_scores = [
        torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)
    ]
    # 第一行全0，取1.0；第二行max是1.5；第三行max是2*1=2.0
    result = compute_adv_scores(turns_tensors, apiswise_adv_scores)
    expected = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)
    assert torch.allclose(result[0], expected), f"Case1 failed: {result[0]} != {expected}"

    # Case 2: 全部不为0
    turns_tensors = [
        torch.tensor([[1,2,3],[1,0,1]], dtype=torch.float32)
    ]
    apiswise_adv_scores = [
        torch.tensor([1.0, 5.0, 2.0], dtype=torch.float32)
    ]
    # max([1*1,2*5,3*2]) = max([1,10,6])=10; max([1*1,0*5,1*2])=2
    result = compute_adv_scores(turns_tensors, apiswise_adv_scores)
    expected = torch.tensor([10.0, 2.0], dtype=torch.float32)
    assert torch.allclose(result[0], expected), f"Case2 failed: {result[0]} != {expected}"

    # Case 3: 行全0
    turns_tensors = [
        torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float32)
    ]
    apiswise_adv_scores = [
        torch.tensor([10.0, 5.0, -1.0], dtype=torch.float32)
    ]
    # 两行都全0，均取1.0（min）
    result = compute_adv_scores(turns_tensors, apiswise_adv_scores)
    expected = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    assert torch.allclose(result[0], expected), f"Case3 failed: {result[0]} != {expected}"

    # Case 4: turns_tensor为None
    turns_tensors = [None]
    apiswise_adv_scores = [torch.tensor([1.0, 2.0, 3.0])]
    result = compute_adv_scores(turns_tensors, apiswise_adv_scores)
    assert result[0] is None, f"Case4 failed: {result[0]} != None"

    print("All tests passed!")

if __name__ == '__main__':
    test_compute_adv_scores()