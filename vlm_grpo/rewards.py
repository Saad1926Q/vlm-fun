import re

from config import REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START


def formatting_reward_func(completions, **kwargs) -> list[float]:
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    scores = []

    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""

        score = 0.0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)

        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0

        # Penalize on excessive addCriterion and newlines
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0
        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    completions = [
        (c[0]["content"] if c else "") if isinstance(c, list) else c
        for c in completions
    ]

    responses = [
        re.findall(answer_pattern, completion, re.DOTALL) for completion in completions
    ]

    return [
        2.0 if len(r) == 1 and a.strip() == r[0].strip() else 0.0
        for r, a in zip(responses, answer)
    ]
