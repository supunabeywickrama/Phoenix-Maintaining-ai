"""
Relevance gate for figures attached to an answer.

Vector search always returns its top-k, so a question about the control panel
still pulls back whatever diagrams the manual happens to have. Showing a
technician the wrong diagram mid-repair is worse than showing none, so figures
get a second check before they reach the answer.

The check is text-only: it judges each candidate by the caption produced at
ingestion rather than re-opening the image, which is one cheap call for the
whole batch instead of one vision call per figure.
"""
from unified_rag.ai_client import chat_json, MODEL_CHAT_LIGHT
from services.llm_json import loads_tolerant

CAPTION_PREVIEW = 400


def verify_images(query: str, images: list, min_keep: int = 0, context_pages: set = None) -> list:
    """Return the subset of `images` genuinely useful for `query`.

    Fails open: if the judge errors or returns nothing usable, the original
    candidates are kept — a degraded judge shouldn't strip a correct diagram.

    `context_pages`: the manual pages the answer's TEXT is actually grounded
    on. Caught a real failure without this — asked about "Clutch Tests" (page
    45, which has no diagram of its own), and the judge kept two Page 46
    "Blade Drive" figures because they are structurally similar (both are
    feeler-gauge/dial-indicator measurement diagrams) even though they
    document a completely different test. Page distance from the answer's own
    grounding is a much harder signal to fool than caption vibes alone.
    """
    if not images or not query.strip():
        return images
    if len(images) == 1 and min_keep >= 1:
        return images

    listing = []
    for i, img in enumerate(images):
        caption = (getattr(img, "content", "") or "").strip().replace("\n", " ")
        listing.append(f"{i}: (page {getattr(img, 'page', '?')}) {caption[:CAPTION_PREVIEW]}")

    pages_note = ""
    if context_pages:
        pages_note = (
            f"\nThe answer's own text is drawn from manual page(s): {sorted(context_pages)}. "
            "A figure from one of these pages, or clearly about the same named test/procedure/"
            "component, is a strong match. A figure from a DIFFERENT page that merely LOOKS "
            "structurally similar (e.g. another test that also uses a feeler gauge or dial "
            "indicator, a different numbered Test/Figure) is very likely a different procedure - "
            "drop it even though it superficially resembles what was asked about.\n"
        )

    prompt = (
        "A maintenance technician asked:\n"
        f"  \"{query}\"\n"
        f"{pages_note}\n"
        "These figures from the machine's manual are candidates to show alongside the answer. "
        "Each line is an index and that figure's description.\n\n"
        + "\n".join(listing)
        + "\n\nReturn JSON: {\"keep\": [indices]} listing ONLY the figures that would actually help "
        "this technician with THIS SPECIFIC question — the exact components, assemblies or "
        "procedures they asked about, not merely a similar-looking test elsewhere in the manual. "
        "Exclude figures about unrelated parts, tests or pages of the machine. If none are "
        "relevant, return an empty array — an empty result is correct when nothing actually matches."
    )

    try:
        raw = chat_json(MODEL_CHAT_LIGHT, prompt, max_tokens=300, temperature=0.0)
        data = loads_tolerant(raw) if raw else None
    except Exception as e:
        print(f"⚠️ [ImageRelevance] Verification failed, keeping candidates: {e}")
        return images

    if not isinstance(data, dict) or "keep" not in data:
        return images

    try:
        keep_idx = {int(i) for i in data["keep"]}
    except (TypeError, ValueError):
        return images

    kept = [img for i, img in enumerate(images) if i in keep_idx]

    if len(kept) < min_keep:
        # The judge was stricter than the caller allows; top up by rank order.
        for img in images:
            if img not in kept:
                kept.append(img)
            if len(kept) >= min_keep:
                break

    dropped = len(images) - len(kept)
    if dropped:
        print(f"🖼️ [ImageRelevance] Dropped {dropped}/{len(images)} figures as off-topic.")
    return kept
