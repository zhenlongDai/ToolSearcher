# import json
# import re
# from typing import Any

# def load_prompt_to_chat_messages(prompt: str, skip_system_message: bool = False,  start_at: int = 0, end_at: int = -1) -> list[dict[str, str]]:

#     message_contents = [e.strip() for e in re.split(r"---+", prompt.strip())]
#     messages: list[dict[str, str]] = []

#     system_name = "system"
#     content_key = "content"
#     user_name = "user"

#     if not  :
#         assert len(message_contents) > 1, "Not enough messages to load in LM."
#         messages = [{author_key: system_name, content_key: message_contents[0]}]
#         message_contents = message_contents[1:]

#     if end_at < 0:
#         end_at = len(message_contents) + end_at + 1
#     for index, message_content in enumerate(message_contents):
#         if index < start_at:
#             continue
#         if index >= end_at:
#             break
#         role = user_name if (index + 1) % 2 == 1 else bot_name
#         messages.append({author_key: role, content_key: message_content})

#     return messages