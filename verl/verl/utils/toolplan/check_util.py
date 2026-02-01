from verl.utils.toolplan.adv_compute import get_count_from_response_mask



def check_turns_data(process_scores, has_answer_state, response_mask):
    response_turns = get_count_from_response_mask(response_mask) 
    caluated_turns = len(process_scores)
    if has_answer_state: 
      caluated_turns += 1

    if response_turns != caluated_turns:
      print(f"response_turns:{response_turns}")
      print(f"caluated_turns:{caluated_turns}")
      return False
    else:
      return True