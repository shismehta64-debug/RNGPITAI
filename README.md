---
title: RNGPIT AI Assistant
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# SINA — RNGPIT AI Assistant

A retrieval-augmented assistant for **R.N.G. Patel Institute of Technology**. It
answers questions about courses, admissions, fees, faculty, facilities,
placements and campus life, in a text chat and in a voice mode with a 3D VRM
avatar.

Built by **Team InnoCrew** — Shis Tushar Maheta, Zuveriya Meman, Karan
Chaudhary, Sem Surti and Shreyansh Vasava.

---

## ⚠️ Read this first if you are upgrading

1. **Rotate your API keys.** The NVIDIA and Supabase keys, and the Flask
   `secret_key`, used to be hard-coded in `app.py`. They are in your git history
   and in every clone. Rotate them, then put the new values in `.env`.
2. **The models changed because the old ones were switched off.**
   `nvidia/nv-embed-v1` reached end of life on 2026-08-25 and
   `mistralai/mistral-medium-3.5-128b` is no longer served, so the app could not
   answer anything at all. Defaults are now `nvidia/nemotron-3-super-120b-a12b`
   (chat) and `nvidia/nemotron-3-embed-1b` (embeddings).
3. **Admin passwords are hashed now.** Existing plaintext rows still log in once
   and are re-hashed automatically, but generate a fresh hash with
   `python manage.py hash-password` and rotate the password.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
python app.py
```

Open <http://localhost:7860>. The admin dashboard is at `/admin/login`.

### Docker

```bash
docker compose up -d --build
```

The compose file keeps `/app/.rngai_cache` in a named volume, so restarts reuse
the vector index instead of paying to rebuild it.

### Configuration

Everything is environment-driven; `.env.example` documents every option. The
ones that matter most:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Flask session signing. Required in production. |
| `NVIDIA_API_KEY` | Chat + embeddings. `NVIDIA_API_KEY_2` adds failover on 429s. |
| `SUPABASE_URL` / `SUPABASE_KEY` | Analytics and admin accounts. Optional — chat works without them. |
| `GROQ_API_KEY` | Voice transcription. Optional. |
| `BEHIND_PROXY` | Set to `1` behind nginx/Cloudflare so `X-Forwarded-For` is trusted. |
| `FORCE_HTTPS` | Set to `1` when served over HTTPS, to add `Secure` to the session cookie. |
| `CORS_ORIGINS` | Only set if the frontend lives on another domain. Empty = same-origin only. |

---

## How it works

```
question
   ├─ greeting / "who made you"? ──────────────► canned answer      (~0 ms, no API calls)
   ├─ seen this question before? ──────────────► response cache     (~0 ms, no API calls)
   │     exact match, or cosine ≥ 0.94 on the query embedding
   └─ otherwise
        ├─ rewrite follow-ups against recent turns
        ├─ expand with domain synonyms (cs → computer science, fees → tuition…)
        ├─ retrieve: BM25 + dense cosine, fused with reciprocal-rank fusion
        ├─ diversify with MMR, pack whole chunks into a character budget
        └─ stream the answer, then cache it
```

### The knowledge base

`data/` is chunked at startup by a structure-aware chunker
(`rngai/chunking.py`) that:

* keeps every Markdown table intact, splitting only oversized ones and
  repeating the header row,
* prefixes each chunk with its heading breadcrumb
  (`Fee Structure & Scholarships > Tuition Fees`) so topic context reaches both
  the embedding and the prompt,
* drops a stale breadcrumb once a heading has "owned" more than ~3,000 tokens,
  because the corpus is scraped page-by-page and a new page often starts at
  `###` without a fresh `#`,
* de-duplicates identical content across files (`link17.txt` was ~45% repeated
  lines).

The result is **667 chunks averaging ~700 characters**, versus 188 chunks
averaging ~3,300 characters before.

The embedded index is written to `.rngai_cache/knowledge_index.json.gz` and
reloaded on boot. It is rebuilt only when the corpus fingerprint or the
embedding model changes.

To add knowledge, drop a `.md` or `.txt` file into `data/` and restart — the
fingerprint changes and the index rebuilds itself.

---

## Operations

```bash
python manage.py check-config                 # what is configured
python manage.py hash-password                # generate an admin password hash
python manage.py rebuild-index --force        # re-embed from scratch
python manage.py ask "what are the CSE fees"  # query the pipeline from the CLI
```

`GET /health` reports index size, cache hit rates, embedding-API call counts and
which integrations are live.

### Tests

```bash
python -m unittest discover -s tests -v
```

80 tests covering chunking, BM25, fusion, vector maths, caching, password
hashing, rate limiting, conversation memory, reasoning-trace filtering, plus
route-level smoke tests (auth, validation, rate limits, security headers). They
need no API keys and no network.

---

## Security notes

* Secrets come only from the environment; nothing is committed.
* Admin passwords are PBKDF2-SHA256 (240k rounds) with constant-time comparison.
  Legacy plaintext and bare-SHA256 rows are accepted once and upgraded in place.
* Every admin endpoint requires authentication; state-changing ones also require
  a double-submit CSRF token. `/api/embeddings/regenerate` and
  `/api/debug/toggle` used to be open to the internet.
* Per-IP rate limits on chat, TTS, transcription, feedback and login.
  `X-Forwarded-For` is only trusted when `BEHIND_PROXY=1`, so the header cannot
  be spoofed for a fresh bucket.
* CORS is off unless `CORS_ORIGINS` is set.
* Model output is rendered through DOMPurify; the previous UI passed
  `marked.parse()` output straight to `innerHTML`, so a document containing HTML
  could run script in every visitor's browser.
* CSP, `X-Frame-Options`, `nosniff` and a `HttpOnly`/`SameSite=Lax` session
  cookie on every response.
* Internal exception text is never returned to clients unless `EXPOSE_ERRORS=1`.

**Known limit:** the rate limiter is in-process. With more than one worker,
limits apply per worker — put a shared limiter in the reverse proxy for a
serious deployment.

---

## The avatar

`static/js/avatar.js` renders the VRM with three.js and `@pixiv/three-vrm`.

* **Lip sync is driven by the audio**, through a WebAudio `AnalyserNode`:
  loudness opens the jaw, spectral centroid selects the vowel blend
  (`aa`/`ih`/`ee`/`ou`/`oh`). It used to be `Math.sin(t * 12)`.
* Motion runs through critically damped springs with breathing, weight shifts,
  gaze saccades, realistic (sometimes double) blinks and secondary arm motion —
  and blends between idle / listening / thinking / speaking instead of snapping.
* The model's chest spring-bone chains (`J_Sec_*_Bust*`) are deleted on load —
  inappropriate for a college assistant, and deleting beats damping because a
  removed joint can never be excited by a pose change or a dropped frame. Hair
  and skirt physics are kept, and the whole spring system is re-initialised at
  the current pose so nothing swings into place when the avatar appears.
* The camera frames itself from the **posed skeleton**, so a replacement VRM
  works without touching the code. (`Box3` is not usable here: on a `SkinnedMesh`
  it measures the bind pose, which for this model is off by the full height of
  the rig.)
* One render loop for the whole page, rendering only the visible tab, pausing
  when the document is hidden, and lowering the pixel ratio when the frame rate
  drops. There were previously four independent `requestAnimationFrame` loops,
  all running forever.

---

## Tech stack

* **Backend:** Flask 3, gunicorn, pure-Python hybrid retrieval
* **Models:** NVIDIA API (Nemotron chat + embeddings), Groq Whisper, Edge TTS
* **Storage:** Supabase (analytics), SQLite (embedding cache), gzip JSON (index)
* **Frontend:** vanilla ES modules, three.js, `@pixiv/three-vrm`, marked, DOMPurify

`torch`, `transformers`, `sentence-transformers` and `chromadb` were removed —
roughly 2.5 GB of wheels that existed to do work the remote API already does. No
local model is ever loaded.

## License

MIT — see [LICENSE](LICENSE).
