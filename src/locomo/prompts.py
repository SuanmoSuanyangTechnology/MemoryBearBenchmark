"""Per-category prompt construction.

All prompt rules are GENERIC task-level rules -- none are keyed to individual
benchmark items.

Selection discipline in SYSTEM (scan all memories, match the question's
focus, DID vs planned), enumerate-then-count for counting stems, multi-event
temporal disambiguation (cat 2), list-inclusion bias (cat 1), and a
commit-dont-refuse rule for cat 1/3/4. Cat 5 keeps its abstention-biased
instruction unchanged.

We reuse LoCoMo's *scoring* unchanged (token-level F1 / refusal detection), so
to stay comparable we mirror LoCoMo's own question conventions from
locomo/task_eval/gpt_utils.py:
  - all categories: short-phrase answers, exact words when possible
        (token-F1 punishes any extra word -> keep it minimal)
  - cat 2 (temporal): nudge the model to answer with an approximate date
  - cat 5 (adversarial): allow refusal; scoring credits ONLY a refusal
        ('no information available' / 'not mentioned')

The only difference vs LoCoMo: the context here is memories retrieved from the
memory system, not raw dialog turns.
"""

SYSTEM = (
    "You answer questions about a long-term conversation between {a} and {b}. "
    "You are given memories retrieved from a memory system. They come from TWO "
    "separate per-person stores, one per speaker, each shown under a "
    "'MEMORY STORE: <name>' header. Within a store, \"the user\" and \"I\" refer to "
    "that store's owner (the name in its header); any other name refers to the other "
    "person. Check which store a memory comes from before attributing a fact to a "
    "person; never credit one person's facts to the other. "
    "Answer ONLY using these memories.\n"
    "Answer format rules:\n"
    "- Always answer in English. If a memory is written in another language, "
    "translate the relevant fact into English; never copy non-English text into "
    "the answer.\n"
    "- Keep the answer short (a single word, a name, a short phrase, or a date) "
    "but COMPLETE: if the answer has several parts or the memories attach "
    "qualifiers to it, include all of them; do not drop items or qualifiers to "
    "save words.\n"
    "- Prefer the most specific detail the memories give (a proper name, a title, "
    "a number, a date) over a generic description, and reuse the memories' exact "
    "English wording whenever possible.\n"
    "- Scan ALL the memories before answering; relevant details are often "
    "scattered, and a memory near the end is as likely to hold the answer as one "
    "near the top. Never settle on the first relevant memory you find.\n"
    "- When several memories relate to the question, pick the one that answers "
    "the question's SPECIFIC focus (what aspect / type / kind is asked), not the "
    "one that merely shares the topic or appears first.\n"
    "- Report what someone actually DID, not what was merely offered, planned, "
    "or available to them.\n"
    "- Do not explain or add a full sentence."
)

def _instr_for(qa):
    """General answer-shape instruction, routed by question type.
    Order matters: adversarial (cat5) is matched first so the numeric/duration
    stem rules never strip its refusal option; then duration/number/frequency
    stems (which cut across categories); then the remaining categories."""
    cat = qa["category"]
    ql = qa["question"].lower().strip()

    # 1) adversarial -- refusal must stay available for every stem
    if cat == 5:
        return ("Answer only if the memories explicitly contain the information the "
                "question asks about; then answer with a short phrase. If the memories "
                "hold only related or similar information, or answering would require "
                "guessing or inference, reply exactly: No information available.")

    # 2) numeric / duration / frequency intents (independent of category)
    if ql.startswith("how long"):
        return "Answer with a duration (e.g. '4 years', 'six months'), NOT a calendar date."
    if ql.startswith(("how many", "how old", "how much")):
        return ("Answer with a single number; count the distinct occurrences in the "
                "memories if needed. First mentally list each distinct instance with "
                "its date or context, merge duplicates (the same event described "
                "twice), then count the list; do not estimate.")
    if ql.startswith("how often"):
        return "Answer with a frequency (e.g. 'twice a week', 'every month')."

    # 3) temporal: a date when a time is asked, otherwise the specific fact
    if cat == 2:
        return ("If the question asks for a time, answer with the date. Resolve relative "
                "expressions against the memory's OWN timestamp: a memory recorded on "
                "date D that says 'yesterday' / 'last week' describes an event BEFORE D, "
                "so answer the resolved time and you may keep the anchor (e.g. 'the week "
                "before 25 May 2023'), not D itself; never answer a bare 'recently "
                "before ...'. Take the year from the memory's timestamp. "
                "If the memories contain SEVERAL similar events at different dates, "
                "mentally list them with their dates before choosing: for a "
                "past-tense question pick the instance closest to (and before) the "
                "question's implied time; for a planned/future event pick the "
                "earliest planned date; never default to the first-mentioned one. "
                "Otherwise answer with the specific fact.")

    # 4) multi-hop: usually an aggregate / list
    if cat == 1:
        return ("If the answer has several parts, gather them from ALL relevant memories "
                "in BOTH stores and list every distinct item (comma-separated) -- "
                "completeness matters more than brevity; otherwise give the single fact. "
                "Include only what the memories support. Before finalizing a list, "
                "re-check for items you were tempted to drop: include every item that "
                "any memory supports; dropping a supported item is worse than including "
                "a marginal one. If the memories contain ANY relevant information, give "
                "your best answer; never reply 'No information available'.")

    # 5) open-domain: allow inference, don't over-refuse
    if cat == 3:
        return ("Answer directly, inferring from the memories when needed (e.g. yes / no / "
                "likely + a brief reason). Do not answer 'No information' if the memories "
                "support a reasonable inference. Follow the direct causal reasoning the "
                "memories suggest; a qualified answer ('likely yes', 'somewhat') is "
                "better than refusing.")

    # 6) single-hop default
    return ("Answer with the specific fact from the memories: the most specific detail "
            "(a proper name, a title, a place, a number), not a generic category, and "
            "keep any qualifiers the memories attach to it (e.g. 'tired but happy', "
            "not just 'tired'). If the memories contain the information, commit to an "
            "answer; reply 'No information available' only when nothing in the "
            "memories bears on the question.")


def build_messages(qa, context, speaker_a, speaker_b):
    """Return OpenAI-style messages for one QA item."""
    instr = _instr_for(qa)
    user = (
        f"Memories about {speaker_a} and {speaker_b}:\n"
        f"{context}\n\n"
        f"Based ONLY on the memories above, {instr}\n\n"
        f"Question: {qa['question']}\nShort answer:"
    )
    return [
        {"role": "system", "content": SYSTEM.format(a=speaker_a, b=speaker_b)},
        {"role": "user", "content": user},
    ]
