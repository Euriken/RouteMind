# Project Spec: Hybrid Token-Efficient Routing Agent
### AMD/Fireworks Hackathon — Track 1 (Beginner) — Team Amavasya (Devansh)

## 1. One-line pitch
"RouteMind" — an autonomous agent that solves any task using the cheapest possible model chain: a local quantized model handles what it can, and escalates to Fireworks AI only when confidence, complexity, or accuracy risk demands it, logging every token to prove it.

## 2. Problem framing
Tasks are unknown until kickoff (July 6), so the agent must be task-agnostic. Local tokens are free. Remote tokens cost. Judged on total token count and output accuracy, with a minimum accuracy floor. Must run in a standardized environment and be fully containerized with Docker.

The real challenge: build a good router, not a good model.

## 3. System architecture
Task in -> Task Classifier (complexity score, task type) -> Semantic Cache lookup (cache hit returns cached answer at 0 tokens) -> Local Model Pass with self-confidence check -> if confident, return local answer at 0 cost -> if not confident, Prompt Compression -> Fireworks AI Call (smallest capable model first, escalate size only if needed) -> Answer + Logger records tokens, route, confidence, latency.

## 4. Core features, build in this order

Tier 1, MVP:
1. Task classifier: heuristic plus local model prompt scoring difficulty 1-5 and type (QA, reasoning, code, summarization, extraction).
2. Local model runner: Ollama serving a small quantized model, candidates qwen2.5:1.5b, phi3:mini, gemma2:2b.
3. Confidence estimator: logprobs if exposed, or self-consistency across 2-3 runs.
4. Router logic: simple decision tree, if confidence >= threshold return local else escalate.
5. Fireworks AI fallback call: API wrapper, retry logic, token accounting from usage field.
6. Token and accuracy logger: JSON/CSV per task with route, tokens, latency, answer.
7. Dockerfile and docker-compose: one command spins up local model server plus orchestrator API.

Tier 2, competitive edge:
8. Semantic caching: embed incoming tasks with all-MiniLM-L6-v2, cosine similarity against past solved tasks, near-duplicate reuses cached answer at zero cost.
9. Prompt compression before escalation: strip boilerplate, summarize long inputs locally first.
10. Model-size escalation ladder: try small Fireworks model first, escalate only if that also fails confidence check.
11. Adaptive threshold tuning: track running accuracy vs token spend, tighten threshold if comfortably above floor, loosen if close to floor.
12. Gemma bonus integration: use Gemma via Fireworks as a rung in the escalation ladder to qualify for Best Use of Gemma Models bonus.

Tier 3, polish:
13. Simple dashboard (Streamlit or Flask+HTML) showing live token spend, route distribution, running accuracy.
14. Batch mode: concurrent processing with rate-limit-aware queuing.
15. Fallback safety net: timeout/error handling, auto-escalate on local model crash.

## 5. Tech stack
Orchestrator API: Flask or FastAPI.
Local model serving: Ollama.
Local model candidates: qwen2.5:1.5b, phi3:mini, gemma2:2b.
Embeddings for cache: sentence-transformers/all-MiniLM-L6-v2, local.
Remote inference: Fireworks AI SDK/REST.
Cache/vector store: FAISS in-memory or SQLite+cosine.
Logging: JSON lines file, optional SQLite.
Containerization: Docker + docker-compose.
Dashboard, optional: Streamlit or small React page.

## 6. Routing decision logic, first version

def route(task):
    cached = semantic_cache.lookup(task)
    if cached and cached.similarity > 0.92:
        return cached.answer, 0

    difficulty = classify_difficulty(task)
    local_answer, confidence = local_model.solve(task)

    if confidence >= THRESHOLD[difficulty]:
        semantic_cache.store(task, local_answer)
        return local_answer, 0

    compressed_prompt = compress(task)
    remote_answer, tokens_used = fireworks_call(compressed_prompt, model="small-tier")

    if remote_confidence_low(remote_answer):
        remote_answer, more_tokens = fireworks_call(compressed_prompt, model="large-tier")
        tokens_used += more_tokens

    semantic_cache.store(task, remote_answer)
    return remote_answer, tokens_used

Tune THRESHOLD[difficulty] empirically once real tasks are seen at kickoff.

## 7. Self-evaluation harness, build before kickoff
Pre-build a generic test harness with placeholder tasks (QA, reasoning, summarization, code mix) to simulate the scoring loop, tune thresholds without wasting Fireworks credits, and stress-test the Docker container end to end. This is the highest-leverage work in the first 24-48 hours.

## 8. 5-Day execution plan, July 6-11

Day 1: get real tasks, understand scoring format. Stand up Docker skeleton, orchestrator plus Ollama container talking to each other. Trivial end-to-end path with no routing logic yet.

Day 2: build classifier plus confidence estimator. Build routing decision function v1 with hardcoded thresholds. Wire Fireworks fallback call.

Day 3: add semantic caching with FAISS. Add prompt compression before escalation. Run self-eval harness against real revealed tasks, log token and accuracy numbers.

Day 4: tune thresholds using Day 3 data. Add model-size escalation ladder plus Gemma bonus integration. Harden error handling, timeouts, retries, local model crash fallback. Finalize Dockerfile/compose, test fresh container build.

Day 5: final testing pass, freeze thresholds. Write README with architecture diagram and routing decision explanation. Record demo and submit. Buffer time for last-minute bugs.

## 9. What to say in the submission write-up
Explain why the local model size was appropriate, not just smallest available. Show the confidence logic behind escalation decisions, not a black box. Show the caching layer's before/after token impact. Call out Gemma usage explicitly for the bonus track.

## 10. Open questions to resolve at kickoff
What exact task types will be revealed. Whether there is a provided scoring script or environment spec. The actual compute limits for the local model. Whether Fireworks usage response gives exact token counts or requires estimation.
