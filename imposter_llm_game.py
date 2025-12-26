import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OPENAI_API_KEY in your .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class Player:
    idx: int
    role: str  # "crewmate" or "imposter"
    knows_word: bool


@dataclass
class GameState:
    secret_word: str
    players: List[Player]
    alive_indices: List[int]
    imposters: List[int]
    crewmates: List[int]
    history: List[Dict]
    votes: List[Dict]


def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 64,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def format_history(history: List[Dict], memory_length: Optional[int]) -> str:
    if not history:
        return "No one has spoken yet."
    if memory_length is not None and memory_length > 0:
        slice_hist = history[-memory_length:]
    else:
        slice_hist = history
    lines = [f"Player {h['player']}: {h['utterance']}" for h in slice_hist]
    return "\n".join(lines)


def generate_clue(
    player: Player,
    game_state: GameState,
    memory_length: Optional[int],
    model: str = DEFAULT_MODEL,
) -> str:
    history_text = format_history(game_state.history, memory_length)

    if player.role == "crewmate":
        system_msg = (
            "You are playing an imposter-style word game. "
            "You KNOW the secret word. "
            "On your turn, you say a short clue (1–3 words) related to the secret word. "
            "Do not say the secret word itself."
        )
        user_msg = (
            f"The secret word is: '{game_state.secret_word}'.\n"
            f"Previous clues so far:\n{history_text}\n\n"
            "Respond with ONLY your clue, no extra text."
        )
    else:
        system_msg = (
            "You are playing an imposter-style word game. "
            "You are an IMPOSTER and you do NOT know the secret word. "
            "Your goal is to avoid being detected. "
            "On your turn, say a short clue (1–3 words) that seems consistent with the previous clues. "
            "Do not admit you don't know the word."
        )
        user_msg = (
            "You do NOT know the secret word.\n"
            f"Previous clues so far:\n{history_text}\n\n"
            "Infer what the word might be and respond with ONLY a plausible clue (1–3 words)."
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return call_llm(messages, model=model, temperature=0.8, max_tokens=16)


def judge_vote(
    game_state: GameState,
    memory_length: Optional[int],
    model: str = DEFAULT_MODEL,
) -> int:
    history_text = format_history(game_state.history, memory_length)
    alive = sorted(game_state.alive_indices)

    system_msg = (
        "You are moderating an imposter word game. "
        "Some players know the secret word, some do not. "
        "Each player has spoken clues. "
        "Your task is to guess which ONE player is most likely an imposter. "
        "You do NOT know the secret word."
    )
    user_msg = (
        f"Players currently alive: {alive}\n"
        "Each line below is 'Player i: clue'. Analyze inconsistencies.\n\n"
        f"{history_text}\n\n"
        "Reply with ONLY the index (integer) of the single player most likely to be an imposter. "
        "No explanation, just the number."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    raw = call_llm(messages, model=model, temperature=0.0, max_tokens=8)

    for tok in raw.replace("\n", " ").split():
        if tok.strip("-+").isdigit():
            try:
                return int(tok)
            except ValueError:
                continue

    return random.choice(alive)


def check_winner(game_state: GameState) -> Optional[str]:
    num_imp = len(game_state.imposters)
    num_crew = len(game_state.crewmates)
    if num_imp == 0:
        return "crewmates"
    if num_imp > num_crew:
        return "imposters"
    return None


def simulate_game(
    secret_word: str,
    N: int = 6,
    m: int = 1,
    rounds_before_vote: int = 1,
    max_rounds: int = 5,
    memory_length: Optional[int] = None,
    vote_memory_length: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    seed: Optional[int] = None,
    imposter_indices: Optional[List[int]] = None,
) -> Dict:
    if seed is not None:
        random.seed(seed)

    all_indices = list(range(N))
    if imposter_indices is None:
        imposter_indices = random.sample(all_indices, m)
    imposters = list(imposter_indices)
    crewmates = [i for i in all_indices if i not in imposters]

    players: List[Player] = []
    for i in all_indices:
        role = "imposter" if i in imposters else "crewmate"
        players.append(Player(idx=i, role=role, knows_word=(role == "crewmate")))

    state = GameState(
        secret_word=secret_word,
        players=players,
        alive_indices=all_indices.copy(),
        imposters=imposters.copy(),
        crewmates=crewmates.copy(),
        history=[],
        votes=[],
    )

    if vote_memory_length is None:
        vote_memory_length = memory_length

    round_num = 0
    winner: Optional[str] = None

    while round_num < max_rounds and winner is None:
        round_num += 1

        for idx in sorted(state.alive_indices):
            player = state.players[idx]
            clue = generate_clue(player, state, memory_length, model=model)
            state.history.append(
                {
                    "round": round_num,
                    "player": idx,
                    "role": player.role,
                    "utterance": clue,
                }
            )

        if round_num % rounds_before_vote == 0:
            voted_idx = judge_vote(state, vote_memory_length, model=model)
            if voted_idx not in state.alive_indices:
                voted_idx = random.choice(state.alive_indices)

            state.alive_indices.remove(voted_idx)
            if voted_idx in state.imposters:
                state.imposters.remove(voted_idx)
            if voted_idx in state.crewmates:
                state.crewmates.remove(voted_idx)

            state.votes.append(
                {
                    "round": round_num,
                    "voted_out": voted_idx,
                    "remaining_imposters": state.imposters.copy(),
                    "remaining_crewmates": state.crewmates.copy(),
                }
            )

            winner = check_winner(state)

    if winner is None:
        winner = "draw"

    return {
        "params": {
            "secret_word": secret_word,
            "N": N,
            "m": m,
            "rounds_before_vote": rounds_before_vote,
            "max_rounds": max_rounds,
            "memory_length": memory_length,
            "vote_memory_length": vote_memory_length,
            "imposter_indices": imposter_indices,
            "model": model,
            "seed": seed,
        },
        "winner": winner,
        "history": state.history,
        "votes": state.votes,
        "final_imposters": state.imposters,
        "final_crewmates": state.crewmates,
    }


if __name__ == "__main__":
    # Tiny smoke test
    res = simulate_game(
        secret_word="piano",
        N=5,
        m=1,
        rounds_before_vote=1,
        max_rounds=3,
        memory_length=10,
        vote_memory_length=10,
        model=DEFAULT_MODEL,
        seed=123,
    )
    print("Winner:", res["winner"])
    print("Final imposters:", res["final_imposters"])
    print("Final crewmates:", res["final_crewmates"])
    print("Votes:", res["votes"])
