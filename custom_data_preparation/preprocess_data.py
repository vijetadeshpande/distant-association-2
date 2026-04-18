import pandas as pd
import numpy as np
import json
import os
import ipdb
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

from system_prompts import SYS_PROMPT, CODENAMES_TASK_INSTRUCTION


def fetch_embedding(word, model):
    """
    Fetch word embedding by assuming that word might have a space character in it
    and averaging the embeddings of the sub-words.
    """
    if word in model:
        return np.array(model[word])
    if " " in word:
        check_str = word.strip().replace(" ", "-")
        if check_str in model:
            return np.array(model[check_str])

    # other cases
    sub_words = word.strip().lower().split(" ")
    assert (type(sub_words) == list) and (len(sub_words) > 0), \
        f"Expected sub_words to be a non-empty list, but got {sub_words}"

    missing_sub_words = []
    found_count = 0
    emb_ = np.zeros_like(model["the"])
    for sub_word in sub_words:
        if sub_word in model:
            emb_ += np.array(model[sub_word])
            found_count += 1
        else:
            missing_sub_words.append(sub_word)

    if found_count == 0:
        #print(f"Warning: No sub-words found in model for '{word}'. Returning zero vector.")
        return emb_

    emb_ = emb_ / found_count  # FIX: average only over found sub-words, not total count
    #if len(missing_sub_words) > 0:
    #    print(f"Warning: The following sub-words were not found in the embedding model "
    #          f"for the word '{word}': {missing_sub_words}")
    return emb_


def calculate_avg_cosine_similarity(sampled_target_set, sampled_non_target_set, model):
    """
    Calculate the average cosine similarity between the sampled_target_set and
    sampled_non_target_set using the provided embedding model.
    """
    emb_non_target = []
    for word in sampled_non_target_set:
        emb = fetch_embedding(word, model)
        if emb is not None:
            emb_non_target.append(emb)
            
    similarities = []
    norm_zero_count = 0
    for target_word in sampled_target_set:
        emb_target = fetch_embedding(target_word, model)
        target_norm = np.linalg.norm(emb_target)
        non_target_norms = np.linalg.norm(emb_non_target, axis=1)

        # FIX: guard against zero-norm vectors causing division by zero
        denom = non_target_norms * target_norm
        if target_norm == 0 or np.any(denom == 0):
            #print(f"Warning: Zero-norm vector encountered for target '{target_word}'. "
            #      "Skipping similarity calculation.")
            norm_zero_count += 1
            continue

        sim = np.dot(emb_non_target, emb_target) / denom
        similarities.append(np.mean(sim))

    if len(similarities) == 0:
        return None, norm_zero_count
    return np.mean(similarities), norm_zero_count


def main():
    word_set_size = 2
    path_glove = "/home/public/vdeshpan/distant-association/custom_data/glove_vectors/dolma_300_2024_1.2M.100_combined.txt"
    path_wordlist = "/home/public/vdeshpan/distant-association/custom_data/wordlists/simple/v3_clustered_words.json"
    path_save = f"/home/public/vdeshpan/distant-association/custom_data/training_prompts/size_{word_set_size}"
    num_examples = 1000

    # read the json file
    with open(path_wordlist, "r") as f:
        words = json.load(f)
    word_categories = list(words.keys())

    # load glove embeddings
    model = {}
    words_w_error = []
    with open(path_glove, "r") as f:
        p_bar = tqdm(total=1287623, desc="Loading GloVe Embeddings...")
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                model[word] = vector
            except ValueError as e:
                # FIX: catch specific exception and safely capture the raw line
                words_w_error.append(line[:50].strip())
            p_bar.update(1)
        p_bar.close()

    # FIX: removed ipdb.set_trace()

    # print some stats about the loaded embeddings
    print("=" * 60)
    print(f"Total words in GloVe model: {len(model)}")
    print(f"Words with loading errors: {len(words_w_error)}")
    print(f"Example words with errors: {words_w_error[:10]}")
    print("=" * 60)

    # @NOTE: this is a trial way of generating data.
    # Super easy board positions with clear separation between words.
    p_bar = tqdm(total=num_examples, desc="Generating Prompts for Training...")
    data = []
    cosine_vals = []
    norm_zero_count_total = 0
    for i in range(num_examples):
        # sample word categories for the board position
        [target_cat, non_target_cat] = np.random.choice(
            word_categories, size=2, replace=False
        ).tolist()
        sampled_target_set = np.random.choice(
            words[target_cat], size=word_set_size, replace=False
        ).tolist()
        sampled_non_target_set = np.random.choice(
            words[non_target_cat], size=word_set_size, replace=False
        ).tolist()

        # calculate the average cosine similarity between the target and non-target sets
        avg_cosine_sim, norm_zero_count = calculate_avg_cosine_similarity(
            sampled_target_set, sampled_non_target_set, model
        )
        if avg_cosine_sim is not None:
            cosine_vals.append(avg_cosine_sim)
            norm_zero_count_total += norm_zero_count

        # generate one train prompt instance
        messages = [{"role": "system", "content": SYS_PROMPT}]
        user_msg = deepcopy(CODENAMES_TASK_INSTRUCTION)
        user_msg = user_msg.replace("[FILL-IN-TARGET-WORDS]", str(sampled_target_set))
        user_msg = user_msg.replace("[FILL-IN-NON-TARGET-WORDS]", str(sampled_non_target_set))
        messages.append({"role": "user", "content": user_msg})

        # one training instance
        instance = {
            "prompt": messages,
            "data_source": "/home/public/vdeshpan/distant-association/custom_reward_functions/codenames_reward.py",
            "reward_model": {"ground_truth": target_cat},
            "extra_info": {
                "task": "codenames",
                "target_words": sampled_target_set,
                "non_target_words": sampled_non_target_set,
                "avg_cosine_similarity": avg_cosine_sim,
            },
            "avg_cosine_similarity": avg_cosine_sim,  # for easy access during sorting
        }
        data.append(instance)
        p_bar.update(1)
    p_bar.close()

    # divide the data into train and validation splits (99-1 split), and order both by avg_cosine_similarity
    df = pd.DataFrame(data)
    train_df = df.sample(frac=0.99, random_state=42)
    val_df = df.drop(train_df.index)
    ipdb.set_trace()
    train_df = train_df.sort_values(by="avg_cosine_similarity", ascending=True).drop(columns=["avg_cosine_similarity"]).reset_index(drop=True)
    val_df = val_df.sort_values(by="avg_cosine_similarity", ascending=True).drop(columns=["avg_cosine_similarity"]).reset_index(drop=True)

    # save the data as parquet file
    os.makedirs(path_save, exist_ok=True)  # FIX: path_save is the dir itself, no need for dirname
    train_df.to_parquet(f"{path_save}/only_codenames_trial_train.parquet", index=False)
    val_df.to_parquet(f"{path_save}/only_codenames_trial_val.parquet", index=False)

    # print basic stats about the generated data
    print("=" * 60)
    print(f"Total training instances generated: {len(data)}")
    print(f"Average cosine similarity across instances: {np.mean(cosine_vals):.4f}")
    print(f"Total zero norm counts across instances: {norm_zero_count_total}")
    print("=" * 60)

    # plot histogram of cosine similarity values
    plt.figure()  # FIX: explicitly create a new figure
    plt.hist(cosine_vals, bins=50)
    plt.title("Distribution of Average Cosine Similarity\nbetween Target and Non-Target Sets")
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{path_save}/cosine_similarity_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()