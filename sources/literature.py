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
import time

import requests
from collections import defaultdict
from dataclasses import dataclass, field

TIMEOUT = 600

# The reader (map) is treated as an optional enrichment, never a dependency.
# Extraction is deterministic and works without it, so a failing reader must cost
# a bounded probe rather than a stalled pipeline: at 60s per paper internally, a
# blind retry over fifty documents is fifty minutes of waiting to learn the same
# thing a single probe learns in twenty seconds.
# The corpus search also fails under sustained load, independently of the query:
# the same pattern that ran six times in a row succeeded, then failed minutes
# later after heavy use. Retries are spaced rather than immediate, since an
# instant retry into a rate limit just spends the attempt.
GREP_ATTEMPTS = 3
GREP_BACKOFF_S = 6

MAP_PROBE_TIMEOUT = 25          # seconds; shorter than the reader's own 60s stall
MAP_HEALTH_TTL = 300            # re-probe at most once every 5 minutes
_map_health: dict[str, float | bool] = {"ok": False, "checked_at": 0.0}
MIN_LEN, MAX_LEN = 15, 120
TOP_N = 10        # best-attested candidates returned to the agent       # shorter is a primer, longer is not an aptamer

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
    paper_affinities: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)

    @property
    def corroboration(self) -> int:
        return len(set(self.papers))


def _run(args: list[str], timeout: int | None = None) -> tuple[str, str, int]:
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
                              text=True, timeout=timeout or TIMEOUT)
    except subprocess.TimeoutExpired:
        return "", f"paperclip {args[0]} exceeded {timeout or TIMEOUT}s", 124
    return proc.stdout, proc.stderr, proc.returncode


def map_healthy(force: bool = False) -> bool:
    """Is the reader answering? Probed cheaply, and the answer cached.

    One paper, a trivial question, and a timeout shorter than the reader's own
    stall. Retrying a dead service is only worth doing if finding out is cheap;
    the expensive mistake is discovering it fifty documents in.
    """
    now = time.monotonic()
    if not force and now - float(_map_health["checked_at"]) < MAP_HEALTH_TTL:
        return bool(_map_health["ok"])

    probe, _, rc = _run(["search", "-s", "pmc", "-n", "1", "aptamer"],
                        timeout=MAP_PROBE_TIMEOUT)
    rid = _results_id(probe)
    ok = False
    if rc == 0 and rid:
        out, _, rc2 = _run(["map", "--from", rid, "-n", "1", "Title?"],
                           timeout=MAP_PROBE_TIMEOUT)
        # A tick means a paper was actually read; "Map complete: 0/1" is a
        # failure that still exits zero, so the return code alone is not enough.
        ok = rc2 == 0 and "\u2713 " in out
    _map_health.update({"ok": ok, "checked_at": now})
    return ok


# paperclip prints its own failures to stdout and still exits 0: a corpus grep
# that dies reports "ERR: ... corpus search error ... [exit 2]" through a
# successful process. Trusting the return code turns that into "no papers
# matched", which is a scientific claim manufactured from a broken query.
_ERR_RE = re.compile(r"^\s*ERR:|corpus search error|\[exit [1-9]", re.M)


def _failed(stdout: str) -> str:
    m = _ERR_RE.search(stdout or "")
    return stdout[m.start():m.start() + 160].strip() if m else ""


def _results_id(text: str) -> str | None:
    m = re.search(r"results_id:\s*(\w+)", text)
    return m.group(1) if m else None


UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
_synonym_cache: dict[str, list[str]] = {}

# Short tokens collide across proteins - HGF is hepatocyte growth factor to most
# readers, whatever else it may abbreviate - so curated names are taken whole and
# no abbreviation is invented from them.
MIN_SYNONYM_LEN = 4
# Long multi-word names bloat the alternation, and the corpus engine fails on an
# over-large boolean pattern - reporting the failure as zero matches. Names above
# this length are dropped; the short specific ones (Cachectin, IFNB2) carry the
# recall benefit anyway.
MAX_SYNONYM_LEN = 26
MAX_VARIANTS = 8

# The corpus engine rejects an over-large boolean pattern with "corpus search
# error", so the alternation is capped by the length it produces rather than by
# how many names it holds. Measured: 269 characters runs, 375 fails. Counting
# variants was the wrong unit - two long synonyms cost more than six short ones.
MAX_ALTERNATION_CHARS = 150


def _gene_candidates(target: str) -> list[str]:
    """Plausible gene symbols for a typed target, most specific first.

    Stripping punctuation is not enough: "TNF-alpha" becomes "TNFalpha", which is
    not a symbol and returns nothing, so the curated synonyms - including
    cachectin - were silently missed for exactly the target that needed them.
    The Greek-letter suffix has to come off as well.
    """
    t = target.strip()
    out = [re.sub(r"[^A-Za-z0-9]", "", t)]
    m = re.match(r"^(.*?)[-_ ]?(alpha|beta|gamma|delta|a|b|\u03b1|\u03b2|\u03b3)$", t, re.I)
    if m and len(m.group(1)) >= 2:
        out.append(re.sub(r"[^A-Za-z0-9]", "", m.group(1)))
    return [x for x in dict.fromkeys(out) if x]


def uniprot_synonyms(target: str) -> list[str]:
    """Curated gene and protein synonyms for a human target, from UniProt.

    Hand-written orthographic rules cover IL-6 / IL6 / interleukin-6 but not what
    a 2008 paper actually called the protein. UniProt lists cachectin for TNF,
    IFNB2 and B-cell stimulatory factor 2 for IL-6, and cytokine synthesis
    inhibitory factor for IL-10 — names an aptamer paper may use throughout
    without ever writing the modern one.
    """
    key = target.upper()
    if key in _synonym_cache:
        return _synonym_cache[key]

    out: list[str] = []
    for gene in _gene_candidates(target):
        if out:
            break
        try:
            r = requests.get(UNIPROT, params={
                "query": f"gene:{gene} AND organism_id:9606 AND reviewed:true",
                "fields": "gene_names,protein_name", "format": "json", "size": 5,
            }, timeout=20)
            r.raise_for_status()
            hits = r.json().get("results", [])
        except (requests.RequestException, ValueError, KeyError):
            continue                          # offline is not an error here
        for hit in hits:
            names = []
            for g in hit.get("genes", []):
                names.append(g.get("geneName", {}).get("value", ""))
                names += [x.get("value", "") for x in g.get("synonyms", [])]
            # Verify by gene name: UniProt ranks loosely, and taking the top hit
            # would attach another protein's synonyms to this search.
            if not any(n.upper() == gene.upper() for n in names):
                continue
            desc = hit.get("proteinDescription", {})
            names.append(desc.get("recommendedName", {})
                         .get("fullName", {}).get("value", ""))
            names += [a.get("fullName", {}).get("value", "")
                      for a in desc.get("alternativeNames", [])]
            out = [n for n in names
                   if MIN_SYNONYM_LEN <= len(n) <= MAX_SYNONYM_LEN]
            break

    _synonym_cache[key] = out
    return out


def _split_target(target: str) -> list[str]:
    """The distinct names inside whatever the agent typed.

    The agent does not pass the string the user entered. It writes things like
    "IL-6 (interleukin-6)" and "IL-6 / interleukin-6", and those went straight
    into the search pattern: the parentheses and slash broke the corpus query
    outright, which surfaced as an intermittent SEARCH FAILED that was in fact
    perfectly deterministic given the phrasing. "human IL-6" did not fail but
    matched three papers instead of fifty-three, which is worse - a quiet loss
    rather than a loud one.

    Each name is pulled out and searched on its own terms.
    """
    parts = re.split(r"[/,;]|\(|\)", target)
    names, seen = [], set()
    for raw in parts:
        name = raw.strip().strip("-· ")
        # Drop qualifiers that are not part of any name a paper would print.
        name = re.sub(r"^(?:human|recombinant|mature|full[- ]length)\s+", "",
                      name, flags=re.I).strip()
        if len(name) >= 2 and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names or [target.strip()]


def _name_variants(target: str) -> list[str]:
    """The spellings a paper might actually use.

    Searching the string the user typed is why the first version of this found
    almost nothing: nobody writes "TNF-alpha" in a paper, they write TNF-alpha
    with a Greek letter, or just TNF. Likewise IL-6 appears as IL6 and as
    interleukin-6. The grep pattern has to carry all of them.
    """
    names = _split_target(target)
    t = names[0]
    out = set()
    for n in names:
        out.update({n, n.replace("-", ""), n.replace("-", " ")})

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

    # Orthographic forms of the name the user typed come first and are never
    # dropped: IL-6, IL6 and interleukin-6 are what most papers actually print,
    # and an earlier cut by descending length discarded exactly those in favour
    # of "CTL differentiation factor". Curated synonyms fill the remaining slots.
    primary = sorted({v.strip() for v in out if len(v.strip()) >= 2},
                     key=len, reverse=True)
    extra = [n for n in uniprot_synonyms(target) if n not in primary]
    return (primary + extra)[:MAX_VARIANTS]


# 5'-GGTGG... or 5′ GG TGG CAG..., with or without spaces between bases.
_PRIMED_RE = re.compile(r"5\s*[\u2032\u2019'`]\s*[-\u2013\u2014]?\s*"
                        r"((?:[ACGTUacgtu][ \t]?){15,})")
_BARE_RE = re.compile(r"(?<![A-Za-z])((?:[ACGTU]){18,})(?![A-Za-z])")
# Affinity is written many ways and rarely sits beside the sequence: sequences
# live in Methods, dissociation constants in Results. The pattern therefore
# covers the common phrasings, and is searched over the whole paper rather than
# the sequence's own paragraph.
_KD_RE = re.compile(
    r"(?:[Kk]\s*[_ ]?\s*[dD]\b|dissociation constant|binding affinit(?:y|ies)|"
    r"affinit(?:y|ies)\s+of)[^.\n]{0,60}?"
    r"([0-9]+(?:\.[0-9]+)?\s*(?:\u00b1\s*[0-9.]+\s*)?"
    r"(?:fM|pM|nM|[u\u00b5]M|mM))")
_NAME_RE = re.compile(r"\b([A-Z][A-Za-z]{1,10}[- ]?\d{1,3}(?:\.\d)?|[A-Z]{2,5}\d{1,3})\b")


SEQ_CONTEXT_PATTERN = "5[\u2032\u2019'`][ -]*(?:[ACGTacgt][ ]?){15,}"
PER_PAPER_CAP = 25       # bound the slow path; partial coverage is reported


def _paragraphs(grep_output: str) -> list[tuple[str, str]]:
    """(doc_id, paragraph) pairs from a corpus grep, which interleaves them."""
    out, doc = [], ""
    for line in grep_output.splitlines():
        m = GREP_DOC_RE.match(line)
        if m:
            doc = m.group(1)
        elif doc and line.strip():
            out.append((doc, line))
    return out


def _sequences_in(para: str, target_re: re.Pattern) -> list[tuple[str, str, str, str]]:
    """Sequences in this paragraph, but only if the target is named in it too."""
    if not target_re.search(para):
        return []
    kd = _KD_RE.search(para)
    kd_text = kd.group(1).strip() if kd else ""
    found = []
    for m in list(_PRIMED_RE.finditer(para)) + list(_BARE_RE.finditer(para)):
        raw = re.sub(r"[^ACGTUacgtu]", "", m.group(1)).upper()
        if not MIN_LEN <= len(raw) <= MAX_LEN:
            continue
        lead = para[max(0, m.start() - 90):m.start()]
        names = _NAME_RE.findall(lead)
        # Keep the sentence fragment the sequence was printed in. A bare string
        # of bases cannot be told apart from a qPCR primer, and the paper almost
        # always says which within a few words of it - "IL-6 adaptor", "forward
        # primer", "selected aptamer". Without it the caller is guessing.
        context = re.sub(r"\s+", " ", lead).strip()[-90:]
        found.append((raw, names[-1] if names else "", kd_text, context))
    return found


def _paper_affinities(doc_id: str) -> list[str]:
    """Every affinity stated anywhere in this paper.

    Paper-level, not sequence-level, and labelled as such downstream. A paper
    reporting a selection round quotes several Kd values for several sequences,
    so pinning one to a particular sequence from a whole-document scan would be a
    guess. Reported so a reader can follow it up, used as an input only when the
    number sits in the same paragraph as the sequence itself.
    """
    out, _, rc = _run(["grep", "-n", "-m", "40",
                       "[Kk][ _]?[dD]|dissociation constant|binding affinit",
                       f"/papers/{doc_id}/content.lines"])
    if rc != 0:
        return []
    return [m.group(1).strip() for m in _KD_RE.finditer(out)]


def _extract_from_paper(doc_id: str, target_re: re.Pattern) -> list[tuple[str, str, str, str]] | None:
    """(sequence, name, kd) triples this paper attributes to the target.

    Deterministic: the paragraph is fetched and parsed here rather than handed to
    a reader model. That is not merely a fallback for when the reader is down. It
    is stricter, because it can insist the target is named in the *same paragraph*
    as the sequence. Document-level co-occurrence, which is all the reader was
    given, is what returned anti-HIV-integrase aptamers for an IL-6 query: those
    papers do mention IL-6, several sections away from the sequence.
    """
    out, _, rc = _run(["grep", "-n", "5[\u2032\u2019'`]|[ACGT]{18,}",
                       f"/papers/{doc_id}/content.lines"])
    if rc != 0:
        return None                      # a failed read, distinct from an empty one
    found: list[tuple[str, str, str]] = []
    for para in out.splitlines():
        found.extend(_sequences_in(para, target_re))
    return found


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

    # The corpus search fails intermittently under load, and a failure that
    # clears on retry should not end the run: the alternative is telling the
    # user nothing was found when the query never executed.
    for attempt in range(GREP_ATTEMPTS):
        grep_out, grep_err, grep_rc = _run(["grep", "--bool", query, "/papers/"])
        printed = _failed(grep_out)
        if grep_rc == 0 and not printed:
            break
        if attempt + 1 < GREP_ATTEMPTS:
            time.sleep(GREP_BACKOFF_S * (attempt + 1))

    if grep_rc != 0 or printed:
        return {"target": target, "searched_as": variants, "n_papers": 0,
                "parents": [], "search_failed": True,
                "search_stage": "corpus query",
                "attempts": GREP_ATTEMPTS,
                "note": f"literature search failed after {GREP_ATTEMPTS} attempts: "
                        f"{printed or grep_err.strip()[:200]}. This is NOT "
                        f"evidence that no aptamer exists — the query itself did "
                        f"not run."}
    rid = _results_id(grep_out)
    n_papers = len(set(GREP_DOC_RE.findall(grep_out)))
    if not rid or not n_papers:
        # The search ran and matched nothing. That is a real answer about this
        # target, distinct from the search having failed, and conflating the two
        # is how an outage became "no IL-6 aptamer exists" earlier.
        return {"target": target, "searched_as": variants, "n_papers": 0,
                "papers_read": 0, "papers_failed": 0, "parents": [],
                "note": "the search ran and no paper contains both a "
                        "target-aptamer phrase and a written-out sequence. This "
                        "is a genuine absence in the corpus, not a failure — "
                        "check the spelling of the target first."}

    # Deterministic extraction. Two passes, because neither call alone is both
    # complete and cheap: the corpus-wide grep answers in one round trip but
    # truncates each paragraph, so a sequence sitting late in a long methods
    # paragraph is cut off — which is exactly where the IL-6 parent lives. The
    # per-paper grep returns whole paragraphs but needs one call per document,
    # and fifty of those in a row is slow and gets throttled.
    target_re = re.compile("|".join(re.escape(v) for v in variants), re.I)
    by_seq: dict[str, Parent] = {}
    order: list[str] = []
    docs = list(dict.fromkeys(GREP_DOC_RE.findall(grep_out)))[:max_papers]
    covered: set[str] = set()

    bulk, _, bulk_rc = _run(["grep", "-n", "-m", "400", SEQ_CONTEXT_PATTERN,
                             "/papers/", "--from", rid])
    if bulk_rc == 0:
        for doc_id, para in _paragraphs(bulk):
            hits = _sequences_in(para, target_re)
            for seq, name, kd, ctx in hits:
                _absorb({"sequence": seq, "aptamer_name": name, "kd": kd,
                         "chemistry": "DNA", "context": ctx},
                        doc_id, by_seq, order)
            if hits:
                covered.add(doc_id)

    # Second pass only where the first found nothing, so the expensive path is
    # bounded by how much the cheap one missed rather than by corpus size.
    probed = failed = 0
    for doc_id in [d for d in docs if d not in covered][:PER_PAPER_CAP]:
        probed += 1
        hits = _extract_from_paper(doc_id, target_re)
        if hits is None:
            failed += 1
            continue
        for seq, name, kd, ctx in hits:
            _absorb({"sequence": seq, "aptamer_name": name, "kd": kd,
                     "chemistry": "DNA", "context": ctx}, doc_id, by_seq, order)

    # One extra pass over just the papers that produced a candidate, to pick up
    # affinities stated outside the sequence's own paragraph.
    # Sorted before slicing. Python randomises string hashing per process, so
    # iterating a set of document ids gives a different order in every run:
    # list(set)[:25] would silently probe a different 25 papers each time and
    # return different affinities for the same query. Reproducible only by
    # accident today, because IL-6 yields fewer papers than the cap.
    seq_papers = sorted({d for p in by_seq.values() for d in p.papers})
    affinity_by_paper = {d: _paper_affinities(d) for d in seq_papers[:PER_PAPER_CAP]}
    for parent in by_seq.values():
        seen_aff: list[str] = []
        for d in sorted(set(parent.papers)):
            for a in affinity_by_paper.get(d, []):
                if a not in seen_aff:
                    seen_aff.append(a)
        parent.paper_affinities = seen_aff

    if by_seq:
        # Corroboration first, then whether an affinity was reported for it.
        # Length was the old tiebreaker and it is meaningless: a 76-mer is not a
        # better parent than a 31-mer for being longer, and ranking on it put an
        # uncharacterised sequence above the one with a sensor built on it.
        parents = sorted(
            (by_seq[k] for k in order),
            key=lambda p: (-p.corroboration, not p.kd_notes, order.index(p.sequence)))
        return {"target": target, "searched_as": variants, "n_papers": n_papers,
                "papers_read": len(covered) + probed - failed,
                "papers_failed": failed,
                "total_sequences_matched": len(parents),
                "extraction": "deterministic regex, target named in the same "
                              "paragraph as the sequence",
                "caveat": "matches are attributed by proximity, not by reading. "
                          "Check that a candidate's paper is really about this "
                          "target before designing from it.",
                "parents": [_as_dict(p) for p in parents[:TOP_N]]}

    # Deterministic extraction found nothing. Only now is the reader worth
    # trying, and only if it is actually answering.
    if not map_healthy():
        return {"target": target, "searched_as": variants, "n_papers": n_papers,
                "papers_read": len(covered) + probed - failed,
                "papers_failed": failed, "parents": [], "search_failed": True,
                "reader_available": False,
                "note": (f"no sequence was extractable from {n_papers} matched "
                         f"papers, and the literature reader is not responding, "
                         f"so the fallback could not run either. This is NOT "
                         f"evidence that no aptamer exists for {target}.")}

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
        ctx = str(entry.get("context", "")).strip()
        if ctx and ctx not in p.contexts:
            p.contexts.append(ctx)
        kd = str(entry.get("kd", "")).strip()
        if kd and kd.lower() not in {"", "none", "not reported", "n/a"} \
                and kd not in p.kd_notes:
            p.kd_notes.append(kd)


_KD_UNITS = {"fM": 1e-6, "pM": 1e-3, "nM": 1.0, "uM": 1e3, "\u00b5M": 1e3, "mM": 1e6}


def parse_kd_nM(text: str) -> float | None:
    """A reported affinity as nanomolar, or None if it cannot be read.

    Papers write affinity a dozen ways - '8.5 +/- 1 nM', '0.4 uM', 'Kd = 209 nM'.
    Leaving it as prose meant the number was visible but unusable, so apparent Kd
    went uncomputed even when the literature had supplied the input it needed.
    """
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:\u00b1\s*[0-9.]+\s*)?"
                  r"(fM|pM|nM|[u\u00b5]M|mM)", text)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    factor = _KD_UNITS.get(unit) or _KD_UNITS.get(unit.replace("u", "\u00b5"))
    return round(value * factor, 4) if factor else None


def _as_dict(p: Parent) -> dict:
    # Take the tightest reported affinity when a paper quotes several: aptamer
    # papers often list a whole selection round, and the lead sequence is the
    # one carried forward.
    parsed = [(parse_kd_nM(k), k) for k in p.kd_notes]
    usable = sorted((v, raw) for v, raw in parsed if v is not None)
    kd_nM, kd_raw = usable[0] if usable else (None, "")

    return {
        "sequence": p.sequence,
        "length": len(p.sequence),
        "chemistry": p.chemistry,
        "corroborating_papers": p.corroboration,
        "papers": sorted(set(p.papers)),
        "reported_names": p.names[:4],
        # What the paper says immediately before the sequence — the only signal
        # that separates an aptamer from a primer without reading the paper.
        "printed_as": p.contexts[:2],
        "reported_kd": p.kd_notes[:4],
        # Ready to hand straight to design_plate, with the paper it came from.
        "kd_nM": kd_nM,
        "kd_as_written": kd_raw,
        "kd_source": sorted(set(p.papers))[0] if kd_nM else "",
        # Affinities stated elsewhere in the same papers. Not attributed to this
        # sequence — a paper may report a whole selection round — but worth
        # surfacing so the number can be checked rather than missed.
        "affinities_reported_in_papers": p.paper_affinities[:6],
    }


def watch_reader(interval: int = 180, tries: int = 40) -> bool:
    """Poll until the reader recovers, printing one line per probe.

    Paperclip's own suggestion when their reader is down. Kept out of the design
    path deliberately: extraction does not need it, so this exists to tell you
    when the enrichment layer is worth using again, not to gate anything on it.
    """
    for i in range(1, tries + 1):
        if map_healthy(force=True):
            print(f"probe {i}: reader RECOVERED", flush=True)
            return True
        print(f"probe {i}: reader still down, retrying in {interval}s", flush=True)
        time.sleep(interval)
    print(f"reader still down after {tries} probes", flush=True)
    return False


if __name__ == "__main__":
    import json
    import sys

    if "--watch" in sys.argv:
        sys.exit(0 if watch_reader() else 1)
    if "--status" in sys.argv:
        ok = map_healthy(force=True)
        print(f"paperclip reader (map): {'responding' if ok else 'NOT responding'}")
        print("extraction does not depend on it; deterministic regex is primary.")
        sys.exit(0 if ok else 1)

    target = next((a for a in sys.argv[1:] if not a.startswith("-")), "TNF-alpha")
    result = find_parents(target)
    print(json.dumps(result, indent=2)[:3000])
