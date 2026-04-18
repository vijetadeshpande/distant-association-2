import json
import os
import pandas as pd
from copy import deepcopy
import numpy as np

from system_prompts import SYS_PROMPT, CODENAMES_CLUE_GEN_INSTRUCTION


def main():
    path_input = "/home/public/vdeshpan/distant-association/custom_data/random_board_pos/trial_board_pos.json"
    path_save = "/home/public/vdeshpan/distant-association/custom_data/random_board_pos"

    with open(path_input, "r") as f:
        board_positions = json.load(f)

    data = []
    for bp in board_positions:
        target_set = bp["target_words"]
        non_target_set = bp["non_target_words"]

        messages = [{"role": "system", "content": SYS_PROMPT}]
        user_msg = deepcopy(CODENAMES_CLUE_GEN_INSTRUCTION)
        user_msg = user_msg.replace("[FILL-IN-TARGET-WORDS]", str(target_set))
        user_msg = user_msg.replace("[FILL-IN-NON-TARGET-WORDS]", str(non_target_set))
        messages.append({"role": "user", "content": user_msg})

        instance = {
            "prompt": messages,
            "data_source": "custom_reward_functions/codenames_reward_model_based.py",
            "reward_model": {"ground_truth": None},
            "extra_info": {
                "task": "codenames_clue_generation",
                "target_words": target_set,
                "non_target_words": non_target_set,
                "all_words": np.random.choice(target_set + non_target_set, size=len(target_set + non_target_set), replace=False).tolist(),
                "intra_target_cosine": bp["intra_target_cosine"],
                "intra_non_target_cosine": bp["intra_non_target_cosine"],
                "avg_target_zipf_freq": bp["avg_target_zipf_freq"],
                "avg_non_target_zipf_freq": bp["avg_non_target_zipf_freq"],
            },
        }
        data.append(instance)

    df = pd.DataFrame(data)
    os.makedirs(path_save, exist_ok=True)
    out_path = os.path.join(path_save, "trial_board_pos.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} instances to {out_path}")


if __name__ == "__main__":
    main()
