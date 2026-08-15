"""Find published aptamers for a biomarker, by regex rather than by relevance.

Topic search is the wrong instrument here. Ranking on "IL-6 aptamer" returns
review articles about SELEX methodology, because that is what the words are
about; the thing actually wanted is a literal string of A, C, G and T sitting in
a methods section. So this greps the corpus body for the sequence itself,
constrained to documents that also talk about the target, and only then spends
an LLM read on the survivors.

    grep  --bool '"<target>.{0,14}aptamer" AND "5.{0,3}-(?:[ACGT][ ]?){15,}"'
    map   the survivors, extracting sequence, name, chemistry and Kd

The optional space inside the sequence pattern is not cosmetic. Papers routinely
print aptamers in triplets - 5'-GG TGG CAG GAG GAC TAT TTA TTT GCT TTT CT-3' -
and a pattern demanding contiguous bases silently misses them. Requiring
contiguity cost the one published IL-6 aptamer that already has a continuous
E-AB sensor built on it.

On TNF-alpha that goes 100 loosely-matching papers -> 11 that write a sequence
out -> the 25-mer VR11, reported identically by four independent groups.

Corroboration is counted rather than assumed. A sequence that several unrelated
papers print the same way is a different proposition from one that appears once,
and transcription errors in aptamer sequences are common enough that the
distinction matters before anything is synthesised.

Deliberately not used: `paperclip filter`. Asked to prune those same 100 papers
it removed all of them in 574 ms, sequences included, and it rewrites the result
set in place so the original is gone. grep -> map skips it.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field

TIMEOUT = 600
MIN_LEN, MAX_LEN = 15, 120       # shorter is a primer, longer is not an aptamer

# grep prints "  PMC12345/"; map prints "    PMC12345 · 1923ms".
GREP_DOC_RE = re.compile(r"^\s*(PMC\d+|bio_\S+|med_\S+|arx_[\d.]+)/", re.M)
MAP_DOC_RE = re.compile(r"^\s*(PMC\d+|bio_\S+|med_\S+|arx_[\d.]+)\s*·", re.M)
JSON_RE = re.compile(r"\{.*\}", re.S)
SEQ_RE = re.compile(r"[ACGTUacgtu]{15,}")

# Each paper answers against this shape, so the sequence arrives as a field
# instead of being fished out of a paragraph the reader model rephrases every run.
# An array, not one field. Papers that report several aptamers are common - a
# selection round publishes a family, a sensor paper prints its probe alongside
# controls - and a single-valued schema makes the reader choose one and discard
# the rest. That silently dropped the IL-6 aptamer with an existing continuous
# E-AB sensor from a paper the grep had correctly found.
SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "has_sequence": {"type": "boolean"},
        "aptamers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aptamer_name": {"type": "string"},
                    "sequence": {"type": "string"},
                    "chemistry": {"type": "string", "enum": ["DNA", "RNA", "unknown"]},
                    "kd": {"type": "string"},
                },
                "required": ["sequence"],
            },
        },
    },
    "required": ["has_sequence"],
})


@dataclass
class Parent:
    """One published aptamer, and how well attested it is."""

    sequence: str
    chemistry: str                      # DNA | RNA
    papers: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    kd_notes: list[str] = field(default_factory=list)

    @property
    def corroboration(self) -> int:
        return len(set(self.papers))


def _run(args: list[str]) -> tuple[str, str, int]:
    """(stdout, stderr, returncode) — all three, deliberately.

    An earlier version returned stdout alone. A failing subprocess then looked
    exactly like a successful one that found nothing, and the caller reported
    "no published sequence found across 53 papers" while the extraction service
    was timing out on every one of them. A tool that cannot tell an empty result
    from a broken one will eventually report the absence of something that is
    there.
    """
    try:
        proc = subprocess.run(["paperclip", *args], capture_output=True,
                              text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "", f"paperclip {args[0]} exceeded {TIMEOUT}s", 124
    return proc.stdout, proc.stderr, proc.returncode


def _results_id(text: str) -> str | None:
    m = re.search(r"results_id:\s*(\w+)", text)
    return m.group(1) if m else None


def _name_variants(target: str) -> list[str]:
    """The spellings a paper might actually use.

    Searching the string the user typed is why the first version of this found
    almost nothing: nobody writes "TNF-alpha" in a paper, they write TNF-alpha
    with a Greek letter, or just TNF. Likewise IL-6 appears as IL6 and as
    interleukin-6. The grep pattern has to carry all of them.
    """
    t = target.strip()
    out = {t, t.replace("-", ""), t.replace("-", " ")}

    greek = {"alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4"}
    m = re.match(r"^(.*?)[- ]?(alpha|beta|gamma|delta)$", t, re.I)
    if m:
        stem, letter = m.group(1), greek[m.group(2).lower()]
        out.update({stem, f"{stem}-{letter}", f"{stem}{letter}"})
    for word, sym in greek.items():
        if sym in t:
            out.add(t.replace(sym, word))
            out.add(t.split(sym)[0].rstrip("- "))

    m = re.match(r"^(?:IL|interleukin)[- ]?(\d+[A-Za-z]?)$", t, re.I)
    if m:
        n = m.group(1)
        out.update({f"IL-{n}", f"IL{n}", f"interleukin-{n}", f"interleukin {n}"})

    return sorted({v.strip() for v in out if len(v.strip()) >= 2}, key=len, reverse=True)


def find_parents(target: str, max_papers: int = 60) -> dict:
    """Published aptamer sequences for `target`, with corroboration counts."""
    variants = _name_variants(target)
    alt = "|".join(re.escape(v) for v in variants)
    # 60 characters, not 14: papers introduce an aptamer at a clause's distance
    # from its target ("an aptamer previously selected against human IL-6"), and
    # a tight window drops them. Widening it took IL-6 from 22 matched papers to
    # 48, and pulled in the one aptamer that already has a continuous E-AB sensor
    # built on it.
    query = (f'"(?:{alt}).{{0,60}}aptamer|aptamer.{{0,60}}(?:{alt})" '
             f'AND "5.{{0,3}}-(?:[ACGTUacgtu][ ]?){{15,}}"')

    grep_out, grep_err, grep_rc = _run(["grep", "--bool", query, "/papers/"])
    if grep_rc != 0:
        return {"target": target, "searched_as": variants, "n_papers": 0,
                "parents": [], "search_failed": True,
                "note": f"literature search failed (exit {grep_rc}): "
                        f"{grep_err.strip()[:200]}. This is NOT evidence that no "
                        f"aptamer exists."}
    rid = _results_id(grep_out)
    n_papers = len(set(GREP_DOC_RE.findall(grep_out)))
    if not rid or not n_papers:
        return {"target": target, "searched_as": variants, "n_papers": 0,
                "parents": [],
                "note": "no paper contains both a target-aptamer phrase and a "
                        "written-out sequence"}

    map_out, map_err, map_rc = _run([
        # Read every matched paper, not a prefix of them. The grep result is
        # ordered by document id, not by which papers actually print a
        # sequence, so truncating the list silently drops the useful ones:
        # at -n 12 of 23, TNF-alpha returned nothing at all.
        "map", "--from", rid, "-n", str(min(n_papers, max_papers)),
        "--output-schema", SCHEMA,
        f"List EVERY explicit nucleotide sequence in this paper for an aptamer "
        f"that binds {target}, including probes used in sensors and any variants "
        f"or truncations. Report each sequence exactly as written, 5' to 3', "
        f"removing spaces. Set has_sequence false only if the paper prints no "
        f"such sequence at all.",
    ])

    # Each paper the reader completes is marked with a tick, each one it fails
    # with a cross. Counting both is the difference between "these papers contain
    # no sequence" and "these papers were never read".
    read_ok = map_out.count("\u2713 ")
    read_failed = map_out.count("\u2717 ")
    parents = [_as_dict(p) for p in _parse(map_out)]

    out = {"target": target, "searched_as": variants, "n_papers": n_papers,
           "papers_read": read_ok, "papers_failed": read_failed,
           "parents": parents}

    if map_rc != 0 or (read_failed and not parents):
        out["search_failed"] = True
        out["note"] = (
            f"the reader failed on {read_failed} of {read_ok + read_failed} papers"
            f"{f' (exit {map_rc}: {map_err.strip()[:120]})' if map_rc else ''}. "
            f"An empty result here is a failed extraction, NOT evidence that no "
            f"aptamer exists for {target}. Re-run before concluding anything.")
    elif read_failed:
        out["note"] = (f"{read_failed} of {read_ok + read_failed} papers could not "
                       f"be read; the parents listed may be incomplete.")
    return out


def _parse(map_out: str) -> list[Parent]:
    """Read the per-paper JSON answers, keeping each sequence's provenance."""
    by_seq: dict[str, Parent] = {}
    order: list[str] = []

    for block in re.split(r"\n\s*\u2713 ", map_out):
        doc = MAP_DOC_RE.search(block)
        payload = JSON_RE.search(block)
        if not doc or not payload:
            continue
        try:
            data = json.loads(payload.group(0))
        except json.JSONDecodeError:
            continue
        if not data.get("has_sequence"):
            continue

        for entry in data.get("aptamers") or []:
            _absorb(entry, doc.group(1), by_seq, order)

    parents = [by_seq[s] for s in order]
    parents.sort(key=lambda p: (-p.corroboration, len(p.sequence)))
    return parents


def _absorb(entry: dict, doc_id: str, by_seq: dict, order: list) -> None:
    """Fold one reported aptamer into the running set."""
    if True:
        raw = re.sub(r"[^ACGTUacgtu]", "", str(entry.get("sequence", ""))).upper()
        if not MIN_LEN <= len(raw) <= MAX_LEN:
            return

        chem = entry.get("chemistry") or ("RNA" if "U" in raw else "DNA")
        seq = raw.replace("U", "T")
        # A poly-T or poly-A run at either end is a surface-attachment spacer,
        # not part of the aptamer. Stripping it lets the same aptamer reported
        # with and without a spacer count as one sequence rather than two.
        seq = re.sub(r"^(?:T{6,}|A{6,})|(?:T{6,}|A{6,})$", "", seq)
        if len(seq) < MIN_LEN:
            return

        p = by_seq.get(seq)
        if p is None:
            p = by_seq[seq] = Parent(sequence=seq, chemistry=chem)
            order.append(seq)
        p.papers.append(doc_id)
        name = str(entry.get("aptamer_name", "")).strip()
        if name and name not in p.names:
            p.names.append(name)
        kd = str(entry.get("kd", "")).strip()
        if kd and kd.lower() not in {"", "none", "not reported", "n/a"} \
                and kd not in p.kd_notes:
            p.kd_notes.append(kd)


def _as_dict(p: Parent) -> dict:
    return {
        "sequence": p.sequence,
        "length": len(p.sequence),
        "chemistry": p.chemistry,
        "corroborating_papers": p.corroboration,
        "papers": sorted(set(p.papers)),
        "reported_names": p.names[:4],
        "reported_kd": p.kd_notes[:4],
    }


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "TNF-alpha"
    result = find_parents(target)
    print(json.dumps(result, indent=2)[:3000])
