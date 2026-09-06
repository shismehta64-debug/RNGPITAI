"""Offline tests for the parts of the pipeline that do not need the network.

Run with:  python -m pytest tests -q     (or)     python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests")

from rngai.bm25 import BM25Index, reciprocal_rank_fusion  # noqa: E402
from rngai.cache import ResponseCache, TTLCache  # noqa: E402
from rngai.chunking import (  # noqa: E402
    chunk_document,
    corpus_fingerprint,
    load_corpus,
    parse_blocks,
)
from rngai.conversations import ConversationStore  # noqa: E402
from rngai.prompts import build_messages, build_search_query, instant_reply  # noqa: E402
from rngai.security import RateLimiter, hash_password, verify_password  # noqa: E402
from rngai.text import (  # noqa: E402
    expand_query,
    is_greeting,
    is_identity_question,
    normalize_query,
    tokenize,
)
from rngai.tts import strip_markdown  # noqa: E402
from rngai.vecmath import Matrix, cosine, normalize  # noqa: E402

SAMPLE = """
# RNGPIT

Intro paragraph about the institute.

## Placements

Placement rate is high.

| Branch | Rate |
| --- | --- |
| CSE | 92.3% |
| Civil | 81.34% |

## Faculty

- Prof. Vivek C. Joshi, vcjoshi@rngpit.ac.in
- Prof. Hardi A. Patel, hapatel@rngpit.ac.in
"""


class TestText(unittest.TestCase):
    def test_normalize_query(self):
        self.assertEqual(normalize_query("  What are the FEES?? "), "what are the fees")
        self.assertEqual(normalize_query(""), "")

    def test_tokenize_drops_stopwords_and_folds_plurals(self):
        tokens = tokenize("What are the placement fees for labs?")
        self.assertIn("placement", tokens)
        self.assertIn("lab", tokens)
        self.assertIn("fee", tokens)
        self.assertNotIn("the", tokens)

    def test_plural_and_singular_agree(self):
        self.assertEqual(tokenize("fees"), tokenize("fee"))
        self.assertEqual(tokenize("placements"), tokenize("placement"))

    def test_expand_query_adds_domain_synonyms(self):
        expanded = expand_query("fees for cse")
        self.assertIn("computer science", expanded)
        self.assertIn("tuition", expanded)

    def test_expand_query_is_a_noop_without_matches(self):
        self.assertEqual(expand_query("tell about xyzzy"), "tell about xyzzy")

    def test_greeting_and_identity_detection(self):
        for text in ("hi", "Hello!", "good morning", "thanks"):
            self.assertTrue(is_greeting(text), text)
        for text in ("what are the fees", "hi how do i apply for cse"):
            self.assertFalse(is_greeting(text), text)
        for text in ("who made you?", "Who built this AI", "what is your name"):
            self.assertTrue(is_identity_question(text), text)
        self.assertFalse(is_identity_question("who is the HOD of IT"))


class TestChunking(unittest.TestCase):
    def test_parse_blocks_types(self):
        kinds = [b.kind for b in parse_blocks(SAMPLE)]
        self.assertIn("heading", kinds)
        self.assertIn("table", kinds)
        self.assertIn("list", kinds)

    def test_tables_are_never_split_across_chunks(self):
        chunks = chunk_document(SAMPLE, source="s.md", target_tokens=40, overlap_tokens=5)
        table_chunks = [c for c in chunks if "| CSE |" in c.text]
        self.assertEqual(len(table_chunks), 1, "the table body was duplicated or split")
        self.assertIn("| Branch | Rate |", table_chunks[0].text)

    def test_heading_breadcrumbs_are_attached(self):
        chunks = chunk_document(SAMPLE, source="s.md")
        placement = next(c for c in chunks if "92.3%" in c.text)
        self.assertEqual(placement.heading_path, ["RNGPIT", "Placements"])
        self.assertTrue(placement.embedding_text.startswith("RNGPIT > Placements"))

    def test_chunks_do_not_straddle_headings(self):
        for chunk in chunk_document(SAMPLE, source="s.md"):
            self.assertNotIn("Placement rate", chunk.text.replace("Placement rate is high.", ""))

    def test_huge_table_splits_with_repeated_header(self):
        rows = "\n".join(f"| Person {i} | Role {i} |" for i in range(200))
        doc = f"# T\n\n| Name | Role |\n| --- | --- |\n{rows}\n"
        chunks = chunk_document(doc, source="t.md", target_tokens=120)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("| Name | Role |", chunk.text)

    def test_corpus_dedupes_identical_documents(self):
        tmp = Path(__file__).parent / "_tmpdata"
        tmp.mkdir(exist_ok=True)
        try:
            (tmp / "a.md").write_text(SAMPLE, encoding="utf-8")
            (tmp / "b.md").write_text(SAMPLE, encoding="utf-8")
            both = load_corpus(tmp)
            (tmp / "b.md").unlink()
            one = load_corpus(tmp)
            self.assertEqual(len(both), len(one), "duplicate document was not de-duplicated")
        finally:
            for path in tmp.glob("*"):
                path.unlink()
            tmp.rmdir()

    def test_fingerprint_is_stable_and_sensitive(self):
        a = chunk_document(SAMPLE, source="s.md")
        b = chunk_document(SAMPLE, source="s.md")
        c = chunk_document(SAMPLE + "\n\nExtra line.\n", source="s.md")
        self.assertEqual(corpus_fingerprint(a), corpus_fingerprint(b))
        self.assertNotEqual(corpus_fingerprint(a), corpus_fingerprint(c))


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "Placement rate for computer science is 92.3 percent",
            "Civil engineering placement rate is 81.34 percent",
            "Prof. Vivek C. Joshi vcjoshi@rngpit.ac.in is the IC HOD",
            "The library is 400 square meters with e-resources",
        ]
        self.index = BM25Index(self.docs)

    def test_finds_rare_literal_token(self):
        results = self.index.search("vcjoshi@rngpit.ac.in")
        self.assertTrue(results)
        self.assertEqual(results[0][0], 2)

    def test_ranks_the_right_branch(self):
        results = self.index.search("civil engineering placement")
        self.assertEqual(results[0][0], 1)

    def test_unknown_terms_return_nothing(self):
        self.assertEqual(self.index.search("quidditch"), [])

    def test_rrf_merges_disjoint_rankings(self):
        fused = reciprocal_rank_fusion([[(1, 0.9), (2, 0.8)], [(2, 12.0), (3, 4.0)]])
        ids = [doc_id for doc_id, _ in fused]
        self.assertEqual(ids[0], 2, "the doc ranked well by both should win")
        self.assertEqual(set(ids), {1, 2, 3})


class TestVecMath(unittest.TestCase):
    def test_normalize_and_cosine(self):
        self.assertAlmostEqual(sum(v * v for v in normalize([3.0, 4.0])), 1.0, places=5)
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0, places=5)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0, places=5)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)

    def test_top_k_orders_by_similarity(self):
        matrix = Matrix([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
        ranked = matrix.top_k([1, 0, 0], 2)
        self.assertEqual(ranked[0][0], 0)
        self.assertEqual(ranked[1][0], 2)

    def test_empty_matrix_is_safe(self):
        self.assertEqual(Matrix([]).top_k([1, 2], 3), [])


class TestCache(unittest.TestCase):
    def test_exact_hit_and_miss(self):
        cache = ResponseCache(max_size=4)
        self.assertIsNone(cache.get("what are the fees"))
        cache.put("What are the fees?", "The fees are X.")
        hit = cache.get("what are the FEES")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], "exact")
        self.assertEqual(hit[0].answer, "The fees are X.")

    def test_semantic_hit(self):
        cache = ResponseCache(max_size=4, threshold=0.9)
        cache.put("what are the fees", "Fees answer", query_vector=[1.0, 0.0, 0.0])
        hit = cache.get("how much are the fees", query_vector=[0.98, 0.02, 0.0])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], "semantic")

    def test_semantic_miss_below_threshold(self):
        cache = ResponseCache(max_size=4, threshold=0.99)
        cache.put("fees", "Fees answer", query_vector=[1.0, 0.0])
        self.assertIsNone(cache.get("hostel", query_vector=[0.0, 1.0]))

    def test_ttl_expiry(self):
        cache = ResponseCache(max_size=4, ttl_seconds=0)
        cache.put("q", "a")
        self.assertIsNone(cache.get("q"))

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("c"))

    def test_empty_answers_are_not_cached(self):
        cache = ResponseCache()
        cache.put("q", "   ")
        self.assertIsNone(cache.get("q"))

    def test_ttl_cache(self):
        cache = TTLCache(ttl_seconds=60)
        cache.put("k", {"v": 1})
        self.assertEqual(cache.get("k"), {"v": 1})
        self.assertIsNone(cache.get("missing"))


class TestSecurity(unittest.TestCase):
    def test_hash_roundtrip(self):
        stored = hash_password("hunter2", rounds=1000)
        valid, rehash = verify_password("hunter2", stored)
        self.assertTrue(valid)
        self.assertTrue(rehash, "a low round count should be flagged for upgrade")
        self.assertEqual(verify_password("wrong", stored), (False, False))

    def test_legacy_plaintext_still_authenticates_but_asks_for_rehash(self):
        valid, rehash = verify_password("plain", "plain")
        self.assertTrue(valid)
        self.assertTrue(rehash)
        self.assertEqual(verify_password("nope", "plain"), (False, False))

    def test_legacy_sha256_hash(self):
        import hashlib

        stored = hashlib.sha256(b"secret").hexdigest()
        valid, rehash = verify_password("secret", stored)
        self.assertTrue(valid)
        self.assertTrue(rehash)

    def test_empty_stored_never_authenticates(self):
        self.assertEqual(verify_password("", ""), (False, False))
        self.assertEqual(verify_password("anything", ""), (False, False))

    def test_rate_limiter_blocks_then_recovers(self):
        limiter = RateLimiter()
        for _ in range(3):
            allowed, _ = limiter.check("ip", 3, 60)
            self.assertTrue(allowed)
        allowed, retry_after = limiter.check("ip", 3, 60)
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)
        allowed, _ = limiter.check("other-ip", 3, 60)
        self.assertTrue(allowed, "limits must be per key")


class TestConversations(unittest.TestCase):
    def test_history_round_trip_and_trimming(self):
        store = ConversationStore(max_turns=2)
        for i in range(5):
            store.append("s1", "user", f"q{i}")
            store.append("s1", "assistant", f"a{i}")
        history = store.history("s1", turns=2)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[-1], ("assistant", "a4"))

    def test_sessions_are_isolated_and_bounded(self):
        store = ConversationStore(max_sessions=2)
        for name in ("a", "b", "c"):
            store.append(name, "user", "hi")
        self.assertEqual(store.size(), 2)
        self.assertEqual(store.history("a", 2), [], "oldest session should be evicted")


class TestPrompts(unittest.TestCase):
    def test_instant_replies(self):
        self.assertIn("Team InnoCrew", instant_reply("who made you?"))
        self.assertIsNotNone(instant_reply("hello"))
        self.assertIsNone(instant_reply("what is the fee for CSE"))

    def test_voice_instant_reply_has_no_markdown(self):
        reply = instant_reply("who built you", voice=True)
        self.assertNotIn("|", reply)
        self.assertNotIn("**", reply)

    def test_follow_up_query_is_rewritten_with_history(self):
        history = [("user", "what is the placement rate for CSE"), ("assistant", "92.3%")]
        self.assertIn("placement rate", build_search_query("what about civil?", history))
        long_q = "tell me everything about the chemical engineering department laboratories"
        self.assertEqual(build_search_query(long_q, history), long_q)

    def test_build_messages_shape(self):
        history = [("user", "hi"), ("assistant", "hello")]
        messages = build_messages("fees?", "CONTEXT BODY", history=history)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertIn("CONTEXT BODY", messages[-1]["content"])

    def test_system_prompt_is_small(self):
        from rngai.prompts import CHAT_SYSTEM_PROMPT

        self.assertLess(
            len(CHAT_SYSTEM_PROMPT),
            2500,
            "the system prompt must stay lean - bulk facts belong in the corpus",
        )


class TestThinkFilter(unittest.TestCase):
    """A model's scratchpad must never reach the user - especially not TTS."""

    def _run(self, tokens):
        from rngai.chat import ThinkFilter

        f = ThinkFilter()
        out = "".join(f.feed(t) for t in tokens)
        return out + f.flush()

    def test_passes_plain_text_through_unchanged(self):
        self.assertEqual(self._run(["Hello ", "world", "!"]), "Hello world!")

    def test_strips_a_complete_think_block(self):
        self.assertEqual(
            self._run(["<think>", "reasoning here", "</think>", "The answer is 92.3%."]),
            "The answer is 92.3%.",
        )

    def test_strips_a_block_split_across_tokens(self):
        tokens = ["Before ", "<th", "ink>", "hidden", "</thi", "nk>", "after"]
        self.assertEqual(self._run(tokens), "Before after")

    def test_unterminated_block_is_dropped_entirely(self):
        self.assertEqual(self._run(["<think>", "still reasoning when truncated"]), "")

    def test_text_after_a_block_survives_flush(self):
        self.assertEqual(self._run(["<think>x</think>ok"]), "ok")

    def test_angle_brackets_that_are_not_tags_survive(self):
        self.assertEqual(self._run(["a < b and c > d"]), "a < b and c > d")


class TestCacheGating(unittest.TestCase):
    """Follow-ups depend on history, so they must bypass the response cache."""

    def test_standalone_question_is_self_contained(self):
        history = [("user", "placement rate for CSE"), ("assistant", "92.3%")]
        query = "What is the fee structure for B.Voc Software Development?"
        self.assertEqual(build_search_query(query, history), query)

    def test_follow_up_is_not_self_contained(self):
        history = [("user", "placement rate for CSE"), ("assistant", "92.3%")]
        self.assertNotEqual(build_search_query("what about civil?", history), "what about civil?")

    def test_first_message_is_always_self_contained(self):
        self.assertEqual(build_search_query("what about civil?", []), "what about civil?")


class TestTTSHelpers(unittest.TestCase):
    def test_strip_markdown(self):
        out = strip_markdown("## Head\n\n**bold** and `code`\n\n| a | b |\n- item")
        for junk in ("##", "**", "`", "|"):
            self.assertNotIn(junk, out)
        self.assertIn("bold", out)
        self.assertIn("item", out)


class TestContextPacking(unittest.TestCase):
    def test_packing_respects_budget_and_keeps_whole_chunks(self):
        from rngai.chunking import Chunk
        from rngai.knowledge import KnowledgeBase, RetrievedChunk

        class Cfg:
            mmr_lambda = 0.7

        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.config = Cfg()
        results = [
            RetrievedChunk(Chunk(id=str(i), text="x" * 100, heading_path=["H"]), score=1.0)
            for i in range(10)
        ]
        context, used = kb.pack_context(results, char_budget=500)
        self.assertLessEqual(len(context), 520)
        self.assertLess(len(used), 10)
        self.assertGreaterEqual(len(used), 1)

    def test_packing_always_returns_something(self):
        from rngai.chunking import Chunk
        from rngai.knowledge import KnowledgeBase, RetrievedChunk

        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.config = type("C", (), {"mmr_lambda": 0.7})()
        results = [RetrievedChunk(Chunk(id="1", text="y" * 5000), score=1.0)]
        context, used = kb.pack_context(results, char_budget=200)
        self.assertTrue(context)
        self.assertEqual(len(used), 1)


class TestRealCorpus(unittest.TestCase):
    """Sanity checks against the shipped data files."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).resolve().parent.parent / "data"
        cls.chunks = load_corpus(cls.data_dir) if cls.data_dir.is_dir() else []

    def test_corpus_loads(self):
        self.assertGreater(len(self.chunks), 100, "the knowledge base looks empty")

    def test_most_chunks_have_a_heading(self):
        with_heading = sum(1 for c in self.chunks if c.heading_path)
        self.assertGreater(with_heading / len(self.chunks), 0.7)

    def test_faculty_emails_are_retrievable_lexically(self):
        index = BM25Index([c.embedding_text for c in self.chunks])
        results = index.search("vcjoshi@rngpit.ac.in", top_k=3)
        self.assertTrue(results, "faculty email should be findable")
        self.assertIn("vcjoshi", self.chunks[results[0][0]].embedding_text.lower())

    def test_placement_question_retrieves_placement_content(self):
        index = BM25Index([c.embedding_text for c in self.chunks])
        results = index.search(expand_query("what is the placement rate for computer science"))
        self.assertTrue(results)
        top_text = " ".join(self.chunks[i].embedding_text.lower() for i, _ in results[:5])
        self.assertIn("placement", top_text)

    def test_no_chunk_is_absurdly_large(self):
        oversized = [c for c in self.chunks if len(c.text) > 12000]
        self.assertEqual(oversized, [], "chunks this large defeat retrieval")


if __name__ == "__main__":
    unittest.main(verbosity=2)
