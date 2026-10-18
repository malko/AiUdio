"""Input-text cleanup (`app/textclean.py`)."""

import pytest

from app.textclean import CleanState, Policy, clean, resolve


def _clean(text, language=None, **policy):
    out, _ = clean(text, Policy(**policy), language)
    return out


def test_off_is_a_no_op():
    raw = "Bonjour 😀  **gras**\x00 \"quote\""
    assert _clean(raw) == raw
    assert _clean(raw, emoji="keep") == raw


def test_light_removes_characters_that_cannot_be_spoken():
    assert _clean("Bon\u00adjour\u200b !", cleanup="light") == "Bonjour!"
    assert _clean("a\x00b\x1fc", cleanup="light") == "abc"
    # Typographic punctuation and stray spaces are folded away.
    assert _clean("« Salut  —  ça   va ? »", cleanup="light") == '" Salut - ça va? "'
    # Sentence punctuation is never touched: the prosody depends on it.
    assert _clean("Un, deux. Trois !", cleanup="light") == "Un, deux. Trois!"


def test_emoji_strip_does_not_glue_words():
    assert _clean("fin😀de phrase", emoji="strip") == "fin de phrase"
    assert _clean("Bonjour 😀 !", emoji="strip") == "Bonjour!"
    assert _clean("😀 Bonjour", emoji="strip") == "Bonjour"


def test_emoji_name_speaks_the_common_ones():
    assert _clean("Bravo 👍", emoji="name", language="fr") == "Bravo bravo"
    assert _clean("Merci 🙏", emoji="name", language="French") == "Merci merci"
    assert _clean("Merci 🙏", emoji="name", language="English") == "Merci thanks"
    # Unknown emojis are dropped rather than spelled out.
    assert _clean("Un 🦑 ici", emoji="name", language="fr") == "Un ici"


def test_emoji_clusters_are_one_unit():
    family = "Famille 👨‍👩‍👧‍👦 ici"          # ZWJ sequence
    assert _clean(family, emoji="name", language="fr") == "Famille ici"
    flag = "Paris 🇫🇷"                        # regional indicator pair
    assert _clean(flag, emoji="strip") == "Paris"
    assert _clean("Top 👍🏽", emoji="name", language="fr") == "Top bravo"   # skin tone
    assert _clean("Top 1️⃣", emoji="name", language="fr") == "Top 1"       # keycap
    # Two emojis in a row are two clusters, not one.
    assert _clean("a😀😀b", emoji="strip") == "a b"


def test_light_keeps_emoji_when_policy_says_so():
    assert _clean("Bonjour 😀 !", cleanup="light") == "Bonjour 😀!"


def test_llm_strips_markdown():
    assert _clean("## Titre\n**gras** et _italique_ et `code`", cleanup="llm") == (
        "Titre\ngras et italique et code"
    )
    assert _clean("Voir [la doc](http://x.y) et ![img](a.png) !", cleanup="llm") == (
        "Voir la doc et!"
    )
    assert _clean("- a\n- b\n> quote\n---\nafter", cleanup="llm") == "a\nb\nquote\n\nafter"


def test_llm_keeps_snake_case_intact():
    assert _clean("utilise my_var_name et _ital_ ici", cleanup="llm") == (
        "utilise my_var_name et ital ici"
    )


def test_fenced_blocks_are_dropped_or_announced():
    text = "Avant.\n```python\nprint('x')\n```\nAprès."
    assert _clean(text, cleanup="llm") == "Avant.\n\nAprès."
    assert _clean(text, cleanup="llm", code_blocks="announce") == (
        "Avant.\n\nCode block.\nAprès."
    )
    assert _clean(text, cleanup="llm", code_blocks="announce", language=None) == (
        "Avant.\n\nCode block.\nAprès."
    )
    fr, _ = clean(text, Policy(cleanup="llm", code_blocks="announce"), "fr")
    assert fr == "Avant.\n\nBloc de code.\nAprès."


def test_fence_state_spans_streamed_chunks():
    """The chat feeds one chunk at a time: an open fence must be remembered."""
    state = CleanState()
    policy = Policy(cleanup="llm", code_blocks="announce")
    out = []
    for part in ["Voici :", "\n```python\n", "print(1)\nprint(2)\n", "```\n", "Et voilà."]:
        text, state = clean(part, policy, "fr", state)
        out.append(text)
    assert [o for o in out if o.strip()] == ["Voici:", "Bloc de code.", "Et voilà."]
    assert state.in_code is False
    # A block that never closes stays silent.
    _, state = clean("```\ndef f():\n", policy, "fr", CleanState())
    assert state.in_code is True
    assert clean("encore du code", policy, "fr", state)[0].strip() == ""


def test_cleanup_is_idempotent():
    source = "## T 😀 **g** `x` [l](u) my_var\n```\ncode\n```\nfin 😀"
    for policy in (Policy(cleanup="light"), Policy(cleanup="llm"),
                   Policy(cleanup="llm", emoji="name", code_blocks="announce"),
                   Policy(emoji="strip")):
        once, _ = clean(source, policy, "fr")
        twice, _ = clean(once, policy, "fr")
        assert once == twice


def test_unknown_policy_values_fall_back_to_the_default():
    policy = Policy(cleanup="nonsense", emoji="maybe", code_blocks="?")
    assert (policy.cleanup, policy.emoji, policy.code_blocks) == ("off", "keep", "drop")


def test_resolve_merges_only_what_the_override_sets():
    base = Policy(cleanup="llm", emoji="name", code_blocks="announce")
    assert resolve(base, None) == base
    assert resolve(base, Policy(emoji="strip")) == Policy(
        cleanup="llm", emoji="strip", code_blocks="announce"
    )


@pytest.mark.parametrize("language", [None, "Auto", "English", "en"])
def test_non_french_languages_use_the_english_names(language):
    out, _ = clean("Bravo 👍", Policy(emoji="name"), language)
    assert out == "Bravo thumbs up"
