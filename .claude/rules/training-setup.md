# Overview
This repository is for a research project. In this research project, we are going to train LLMs with RLVR (reinforcement learning with verifiable reward) method provided under the VeRL (Volcano Engine Reinforcement Learning) library. Specifically, we are going to focus on one method DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) and train LLMs with it. The training prompts will be focused a wordgame known as Codenames (refer to @.claude/rules/codenames.md), so, we train the LLMs to learn and get better at codenames. 

## Some helpful resources
- VeRL library - https://github.com/verl-project/verl

## The exact hypothesis of the research project

% Paragraph-1: LLMs have gaps in concept understanding, how can we improve it?
In recent years, large language models (LLMs) have seen exponential growth in capabilities, accompanied by widespread adoption for assistant-like tasks such as writing, learning, and coding. Given this rapid progress, one might conclude that LLMs possess a robust representation of basic world concepts. Indeed, several studies provide positive evidence to this effect, showing that LLMs encode coherent representations. However, this conclusion is far from settled. A growing body of work reveals significant gaps, highlighting a lack of understanding of simple concepts and limited conceptual mapping in text generation. In light of such juxtaposed evidence, we revisit a fundamental question: \textit{How can we improve the existing conceptual understanding of LLMs?}


% Paragraph-2: Associative learning can help, but distant association is a challenge
To address this question, we first consider what drives conceptual learning in LLMs. Multiple independent studies converge on a common finding: LLMs learn primarily through association, with their conceptual knowledge largely shaped by co-occurrence statistics in pretraining data. This associative learning, while a limited form of learning, proves remarkably effective for developing conceptual mappings in LLMs. Yet associative learning (or co-occurrence-based learning) has a well-known failure mode: representational quality degrades when the underlying statistics are sparse. This manifests along two axes. First, \emph{absolute frequency}: rare concepts are poorly represented, leading to degraded performance on infrequent entities and long-tail phenomena. Second, \emph{relative co-occurrence}: concepts that rarely appear together, even if individually frequent, lack the distributional signal needed for models to learn meaningful cross-domain or distant associations. We target these failure modes in our work to improve the conceptual understanding of LLMs.

% Paragraph-3: Codenames simulates distant association
These exact weak links, \emph{rare and distant} concepts, can systematically be focused on with the board game Codenames\footnote{\url{https://en.wikipedia.org/wiki/Codenames_(board_game)}}
In Codenames, a grid of words is divided into sets belonging to two competing teams, with additional neutral and ``assassin'' words. Each team has two roles: a \emph{Spymaster}, who can see which words belong to which team, and an \emph{Operative}, who cannot. The Spymaster must produce a single-word clue that connects as many of their team's words as possible while avoiding association with the opponent's words. The Operative then uses this clue to guess which words on the board belong to their team. We leverage this structure to naturally force the LLM to reason over distant associations. While in the Spymaster role, the LLM must identify non-obvious shared properties across semantically unrelated words (e.g., linking \emph{lens} and \emph{tire} through circularity), while simultaneously distinguishing these from nearby distractors. When an LLM plays both roles, each game instance becomes a structured exercise in discovering and verifying distant conceptual links, providing a scalable source of training signal for the associations that standard pretraining corpora underrepresent.

% Paragraph-4: Game + RL has been a successful recipe and suitable for our application
Codenames offers further practical advantages as a training environment. First, it provides a scalable source of data where rare and distant concepts can naturally co-occur, addressing the distributional gaps identified above. Second, game outcomes can be verified by simple rules, eliminating the need for human feedback. Together, these properties make Codenames well-suited for Self-Play, a training paradigm in which the model generates its own data and learns from rule-based rewards. Self-Play has a strong track record in game-playing agents and scientific discovery. More recently, adapted this paradigm for LLMs under the name Reinforcement Learning with Verifiable Rewards (RLVR). Several subsequent works have demonstrated the effectiveness of RLVR when combined with games. We follow this line of work: we formulate Codenames as a Markov Decision Process and train LLMs using RLVR. However, while prior work has focused on improving general reasoning ability through game-based RLVR, the effect of such training on the model's underlying conceptual representations remains unexplored. This is the central question of our work.

# LLM training setup
Following is a rough sketch of how LLM training wil take place. 

- We assume the training prompts are already available for DAPO training (for exact prompt refer to @custom_data_preparation/system_prompts.py)
- [actor-rollout] - The LLM being trained will generate responses to the training prompts in default single-turn rollout way.
- [reward-calculation] - The brief idea for implementing custom rewards for this task is as follows,
    - [parsing] - The LLM generated responses will be parsed to extract a [Clue] and a list of [Selected-Targets]
    - [guess-prompt] - Based on [All-Words] (all words on the board), [Clue] and [Max-Guesses] (len([Selected-Targets])), we form a prompt for the guess-generation prompt
    - [judge-inference] - We will use the guess generation prompts to perform inference on a Judge model (stronger model than the one being trained)
    - [parsing] - From the responses of the judge model we will extract [Guesses]
    - [reward-calculation] - Based on the [Guesses], [Target-Set] and [Non-Target-Set], we will calculate scalar reward
    - [return] - return the scalar reward
- All the following steps in the DAPO (or GRPO) set-up will be unchanged (i.e., [advantage-calculation] -> [objective-calculation] -> [parameter-update])

***Note:*** There are additional notes on VeRL-specific reward function implementation in @.claude/rules/model-based-reward.md