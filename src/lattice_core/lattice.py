"""General-purpose LATTICE retrieval — run the tree-traversal search on ANY query.

This is the task-agnostic core that `src/run.py` (BRIGHT evaluation), the ECtHR evaluator
(`src/llm_rl_playground/ecthr_evaluation.py`), and the distillation tracer all delegate to
for the actual tree traversal. Use it directly when you just want ranked documents back for a
query, with:
  - no task            (works on any semantic tree),
  - no gold labels     (nothing is scored; `gold_paths` stays empty),
  - no metrics / wandb (pure inference).

`traverse_samples_async` is the single LATTICE loop for the whole codebase; the eval scripts wrap
it with their own gold metrics / logging (and step it one iteration at a time to interleave those).
Response parsing includes the local-model JSON repair (repair + single-candidate fallback + graceful
beam drop) so flaky local/vLLM output can't poison a beam.

--------------------------------------------------------------------------------
Library use
--------------------------------------------------------------------------------
    from lattice_core import HyperParams, LatticeRetriever

    hp = HyperParams.from_args("--subset biology --tree_version bottom-up "
                               "--llm_api_backend openai --llm gpt-4.1")
    retriever = LatticeRetriever.from_hp(hp)

    results = retriever.retrieve("What regulates the lac operon?", num_iters=8, top_k=10)
    for r in results:
        print(r["rank"], round(r["score"], 3), r["node_id"], r["text"][:120])

    # Batch many queries through one shared LLM batch per iteration:
    batch = retriever.retrieve_many(["query A", "query B"], num_iters=8, top_k=10)

--------------------------------------------------------------------------------
CLI use
--------------------------------------------------------------------------------
    python src/lattice.py \
        --subset biology --tree_version bottom-up \
        --llm_api_backend openai --llm gpt-4.1 \
        --num_iters 8 --top_k 10 \
        --query "What regulates the lac operon?"

    # Many queries (one per line), results written as JSON:
    python src/lattice.py --subset biology --tree_version bottom-up \
        --queries_file my_queries.txt --output results.json

    # Point at an arbitrary tree pickle, and define what "relevant" means for it:
    python src/lattice.py --tree_path corpora/MyCorpus/main/trees/tree-bottom-up-llm.pkl \
        --subset mycorpus --tree_version custom \
        --relevance_definition "A document is relevant if it answers the question." \
        --query "..."
"""

#region Imports
import os
import re
import sys
import json
import logging
import asyncio
import argparse
import pickle as pkl
from typing import Any

import numpy as np

from ._async import run_coro_sync
from .hyperparams import HyperParams
from .tree_objects import SemanticNode, InferSample
from .llm_apis import GenAIAPI, LocalModelAPI, VllmAPI, OpenAIResponsesAPI
from .prompts import get_traversal_prompt_response_constraint
from .utils import setup_logger, compute_node_registry, post_process

np.random.seed(42)
# .../src/lattice_core/lattice.py -> repo root is three levels up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#endregion


#region Shared setup (mirrors src/run.py so behavior matches the eval scripts)
def load_semantic_tree(hp: HyperParams, tree_path: str | None = None):
    """Load a semantic tree and its node registry.

    `tree_path` overrides the default
    `corpora/{DATASET}/{SUBSET}/trees/tree-{TREE_VERSION}.pkl`.
    """
    if tree_path is None:
        tree_path = f"{BASE_DIR}/corpora/{hp.DATASET}/{hp.SUBSET}/trees/tree-{hp.TREE_VERSION}.pkl"
    tree_dict = pkl.load(open(tree_path, "rb"))
    root = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
    node_registry = compute_node_registry(root)
    return root, node_registry


def build_llm_api_and_kwargs(hp: HyperParams, logger):
    """Build the LLM API + per-backend run kwargs, identical to `src/run.py`."""
    backend = hp.LLM_API_BACKEND
    if backend == "genai":
        llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
    elif backend == "vllm":
        llm_api = VllmAPI(
            hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES,
            base_url=",".join([f"http://localhost:{8000 + i}/v1" for i in range(4)]), enable_thinking=False,
        )
    elif backend == "openai":
        llm_api = OpenAIResponsesAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
    elif backend in {"local", "localModel"}:
        llm_api = LocalModelAPI(
            hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES,
            adapter_path=hp.LOCAL_ADAPTER_PATH, use_4bit=hp.LOCAL_USE_4BIT,
            serialize_requests=hp.LOCAL_SERIALIZE_REQUESTS,
        )
    else:
        raise ValueError(f"Unknown LLM API backend: {backend}")

    kwargs: dict[str, Any] = {
        "max_concurrent_calls": hp.LLM_MAX_CONCURRENT_CALLS,
        "response_mime_type": "application/json",
        "response_schema": get_traversal_prompt_response_constraint(bool(hp.REASONING_IN_TRAVERSAL_PROMPT)),
        "staggering_delay": hp.LLM_API_STAGGERING_DELAY,
    }
    if backend == "genai":
        from google.genai import types
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=hp.REASONING_IN_TRAVERSAL_PROMPT)
    elif backend == "vllm":
        kwargs.pop("response_mime_type")
        kwargs.pop("response_schema")
    elif backend == "openai":
        kwargs.pop("response_mime_type")
    elif backend in {"local", "localModel"}:
        kwargs.pop("response_mime_type")
        if hp.LOCAL_SERIALIZE_REQUESTS:
            kwargs["max_concurrent_calls"] = 1

    return llm_api, kwargs


def override_relevance_definition(definition: str) -> None:
    """Make the traversal prompt use a custom notion of "relevant" for any tree.

    `get_traversal_prompt` looks up `get_relevance_definition(hp.SUBSET)` and, for an
    unknown subset, silently falls back to the `stackexchange` definition. When you run
    LATTICE on your own corpus this is usually the wrong text, so we patch the lookup to
    return your definition regardless of subset. Leaves run.py / ECtHR behavior untouched
    since they never call this.
    """
    from . import prompts

    prompts.get_relevance_definition = lambda subset: definition
#endregion


#region The general retriever
class LatticeRetriever:
    """Task-free, gold-free LATTICE search over a fixed semantic tree.

    Construct once (loads the tree + LLM API), then call `retrieve` / `retrieve_many`
    as many times as you like.
    """

    def __init__(self, semantic_root_node, node_registry, hp, logger, llm_api, llm_api_kwargs,
                 show_error_logs: bool = False):
        self.semantic_root_node = semantic_root_node
        self.node_registry = node_registry
        self.hp = hp
        self.logger = logger
        self.llm_api = llm_api
        self.llm_api_kwargs = llm_api_kwargs
        # When False, parsing repairs/drops bad responses silently. Set True to see why.
        self.show_error_logs = show_error_logs

    @classmethod
    def from_hp(cls, hp: HyperParams, logger=None, tree_path: str | None = None,
                relevance_definition: str | None = None,
                show_error_logs: bool = False) -> "LatticeRetriever":
        logger = logger or _default_logger()
        if relevance_definition:
            override_relevance_definition(relevance_definition)
        root, node_registry = load_semantic_tree(hp, tree_path=tree_path)
        llm_api, llm_api_kwargs = build_llm_api_and_kwargs(hp, logger)
        return cls(root, node_registry, hp, logger, llm_api, llm_api_kwargs,
                   show_error_logs=show_error_logs)

    def make_sample(self, query: str) -> InferSample:
        """One search state for `query`, with no gold and nothing excluded."""
        return InferSample(
            self.semantic_root_node,
            self.node_registry,
            hp=self.hp,
            logger=self.logger,
            query=query,
            gold_paths=[],
            excluded_ids_set=set(),
        )

    # --- local-model JSON repair (ported from EcthrTraversalEvaluator) ---
    # Local / vLLM backends often emit traversal JSON that won't parse cleanly. Rather
    # than let one bad response poison a beam, we (1) try post_process/repair, (2) fall
    # back to a non-promoting score when the slate has a single candidate, and only then
    # (3) drop it by returning None -- which `sample.update` handles by pruning that beam.

    @staticmethod
    def _valid_candidate_ids_from_prompt(prompt: Any) -> list[int]:
        """Recover the slate's valid candidate IDs from the prompt text (e.g. '0, 1, 2')."""
        if prompt is None:
            return []
        if isinstance(prompt, list):
            prompt_text = "\n".join(str(message.get("content", message)) for message in prompt)
        else:
            prompt_text = str(prompt)
        match = re.search(r"Valid candidate IDs for this request:\s*([^\n.]+)", prompt_text)
        if not match:
            return []
        return [int(candidate_id) for candidate_id in re.findall(r"\d+", match.group(1))]

    @staticmethod
    def _single_candidate_fallback_response(prompt: Any, output: Any) -> dict | None:
        """If the slate had exactly one candidate, synthesize a non-promoting (score 0) response."""
        valid_candidate_ids = LatticeRetriever._valid_candidate_ids_from_prompt(prompt)
        if len(valid_candidate_ids) != 1:
            return None
        candidate_id = valid_candidate_ids[0]
        preview = str(output).replace("\n", " ")[:500]
        return {
            "reasoning": (
                "Local model did not return the required traversal JSON. "
                f"Using the only valid candidate ID {candidate_id} as a non-promoting fallback. "
                f"Raw response preview: {preview}"
            ),
            "ranking": [candidate_id],
            "relevance_scores": [[candidate_id, 0]],
        }

    def _parse_traversal_response_or_none(
        self, output: Any, *, prompt: Any = None, step: int | None = None, prompt_index: int | None = None,
    ) -> dict | None:
        """Repair one traversal response if possible; return None if it is still unusable."""
        parsed = post_process(output, return_json=True, show_error_logs=self.show_error_logs)
        if isinstance(parsed, dict):
            return parsed

        fallback = self._single_candidate_fallback_response(prompt, output)
        if fallback is not None:
            return fallback

        if self.show_error_logs:
            location = []
            if step is not None:
                location.append(f"iteration {step + 1}")
            if prompt_index is not None:
                location.append(f"prompt {prompt_index}")
            location_text = f" ({', '.join(location)})" if location else ""
            preview = str(output).replace("\n", " ")[:500]
            self.logger.warning(
                "Traversal response is badly formatted after attempted repair%s; "
                "dropping it from this update. Raw preview: %s", location_text, preview,
            )
        return None

    async def _parse_traversal_responses_async(
        self, raw_responses: list[Any], *, prompts: list[Any] | None = None, step: int | None = None,
    ) -> list[dict | None]:
        """Parse/repair a whole batch of responses concurrently in worker threads."""
        parse_concurrency = int(self.llm_api_kwargs.get("max_concurrent_calls") or 16)
        semaphore = asyncio.Semaphore(max(parse_concurrency, 1))

        async def parse_one(idx: int, output: Any) -> dict | None:
            async with semaphore:
                prompt = prompts[idx] if prompts is not None and idx < len(prompts) else None
                return await asyncio.to_thread(
                    self._parse_traversal_response_or_none,
                    output, prompt=prompt, step=step, prompt_index=idx,
                )

        return list(await asyncio.gather(*[parse_one(idx, output) for idx, output in enumerate(raw_responses)]))

    async def traverse_samples_async(self, samples: list[InferSample], num_iters: int) -> int:
        """Run the LATTICE loop over pre-built samples: one shared LLM batch per iteration.

        This is the single LATTICE traversal loop for the whole codebase -- `run.py`, the ECtHR
        evaluator, and the distillation tracer all delegate here and add their own gold metrics /
        logging on top. A sample that has exhausted its beam contributes zero prompts and is
        skipped. `retrieve*` build the samples for you; tracer subclasses (see
        `TracingLatticeRetriever`) call it directly to observe each step via `_record_step`.

        Returns the number of iterations that actually ran work (fewer than num_iters if every
        sample exhausts its beam early). Callers that step one iteration at a time (num_iters=1)
        use this to detect exhaustion without a second, RNG-perturbing `get_step_prompts` call.
        """
        steps_run = 0
        for step in range(num_iters):
            inputs = [sample.get_step_prompts() for sample in samples]
            indptr = np.cumsum([0, *[len(x) for x in inputs]])
            flat_inputs = [item for sample_inputs in inputs for item in sample_inputs]
            if not flat_inputs:
                break  # every sample has reached leaves; nothing left to expand

            flat_prompts, flat_slates = list(zip(*flat_inputs))
            flat_responses = await self.llm_api.run_batch(list(flat_prompts), **self.llm_api_kwargs)
            flat_response_jsons = await self._parse_traversal_responses_async(
                list(flat_responses), prompts=list(flat_prompts), step=step,
            )
            skipped = sum(response_json is None for response_json in flat_response_jsons)
            if skipped and self.show_error_logs:
                self.logger.warning(
                    "Dropped %d/%d traversal response(s) still unusable after repair.",
                    skipped, len(flat_response_jsons),
                )

            # Reshape this step's flat batch into a per-sample view, then hand it to the
            # `_record_step` seam. Pure read-only bookkeeping with no side effects, and the seam
            # is a no-op in the base class -- so ordinary retrieval is unaffected. It exists only
            # so tracer runs (e.g. teacher distillation) can observe every
            # (prompt, slate, raw_response, parsed_response) without re-implementing this loop.
            per_sample = []
            for j in range(len(samples)):
                lo, hi = int(indptr[j]), int(indptr[j + 1])
                per_sample.append(list(zip(
                    flat_prompts[lo:hi], flat_slates[lo:hi], flat_responses[lo:hi], flat_response_jsons[lo:hi],
                )))
            self._record_step(step, samples, per_sample)

            for j, sample in enumerate(samples):
                slates = list(flat_slates[indptr[j]:indptr[j + 1]])
                response_jsons = list(flat_response_jsons[indptr[j]:indptr[j + 1]])
                if slates:
                    sample.update(slates, response_jsons)
            steps_run += 1

        return steps_run

    def _record_step(self, step: int, samples: list[InferSample], per_sample: list[list[tuple]]) -> None:
        """Per-step extension point -- a no-op in the base class, meant ONLY for tracer runs.

        This is not part of normal retrieval; `retrieve()`, `run.py`, and the ECtHR evaluator
        never need it. It is called once per iteration, after parsing and before `sample.update`,
        so that tracer subclasses (e.g. teacher distillation, which must log every decision the
        model makes, not just the final ranked leaves) can record each step in place. `per_sample[j]`
        is the list of `(prompt, slate, raw_response, parsed_response)` tuples produced for sample
        `j` this step (empty if that sample is already finished). The base loop ignores the return
        value, so leaving this empty changes nothing for any non-tracing caller.
        """
        return

    @staticmethod
    def print_frontier(sample: InferSample) -> None:
        """Print a sample's current beam frontier (handy under detailed tracer logging)."""
        print("Current beam state paths:")
        for state_path in sample.beam_state_paths:
            node = state_path[-1]
            print(f"  path={node.path} | path_relevance={node.path_relevance:.3f} | desc={node.desc[:180]}")

    @staticmethod
    def _read_top(sample: InferSample, top_k: int) -> list[dict]:
        """Read out the top leaf predictions as plain dicts (native LATTICE ranking)."""
        rows = []
        for rank, (node, score) in enumerate(sample.get_top_predictions(k=top_k), start=1):
            rows.append({
                "rank": rank,
                "score": float(score),
                "node_id": node.id,
                "path": list(node.path),
                "path_relevance": float(node.path_relevance),
                "local_relevance": float(node.local_relevance),
                "text": node.desc,
            })
        return rows

    # --- async API (use inside an existing event loop / notebook) ---
    async def retrieve_many_async(self, queries: list[str], num_iters: int = 8, top_k: int = 10) -> list[list[dict]]:
        samples = [self.make_sample(q) for q in queries]
        await self.traverse_samples_async(samples, num_iters)
        return [self._read_top(s, top_k) for s in samples]

    async def retrieve_async(self, query: str, num_iters: int = 8, top_k: int = 10) -> list[dict]:
        return (await self.retrieve_many_async([query], num_iters=num_iters, top_k=top_k))[0]

    # --- sync API (use from a plain script; safe inside notebooks too) ---
    def retrieve_many(self, queries: list[str], num_iters: int = 8, top_k: int = 10) -> list[list[dict]]:
        return run_coro_sync(self.retrieve_many_async(queries, num_iters=num_iters, top_k=top_k))

    def retrieve(self, query: str, num_iters: int = 8, top_k: int = 10) -> list[dict]:
        return run_coro_sync(self.retrieve_async(query, num_iters=num_iters, top_k=top_k))
#endregion


#region Tracing subclass (for distillation / per-decision recording)
class TracingLatticeRetriever(LatticeRetriever):
    """LATTICE retriever that records every model decision as it traverses.

    Use this for TRACER RUNS such as teacher distillation, where you need the raw and parsed
    response for each (case, step, prompt) -- not just the final ranked leaves. It overrides only
    the `_record_step` seam; the traversal itself is the unmodified base loop, so the parsed
    responses (and therefore any distilled training rows) come out identical to a normal run.

    `case_indexes[j]` is the external id stamped on sample `j`'s rows (e.g. the ECtHR dataset case
    index). After `traverse_samples_async(samples, num_iters)`, read the collected `self.trace_rows`.
    """

    def __init__(self, *args, case_indexes: list[int], detailed_logs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_indexes = case_indexes
        self.detailed_logs = detailed_logs
        self.trace_rows: list[dict] = []

    def _record_step(self, step, samples, per_sample):
        teacher_model = getattr(self.llm_api, "model_name", None)
        for j, sample in enumerate(samples):
            for prompt, slate, raw_response, parsed_response in per_sample[j]:
                self.trace_rows.append({
                    "case_index": self.case_indexes[j],
                    "step": step,
                    "prompt": prompt,
                    "slate": slate,
                    "raw_response": raw_response,
                    "parsed_response": parsed_response,
                    "valid": parsed_response is not None,
                    "query": sample.query,
                    "teacher_model": teacher_model,
                })
            # Frontier here is the beam entering this step (pre-update); it's debug-only output.
            if self.detailed_logs and per_sample[j]:
                self.print_frontier(sample)
#endregion


#region CLI
def _default_logger():
    logger = logging.getLogger("lattice")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _parse_cli(argv=None):
    """Peel off lattice-only flags, then let HyperParams parse the rest (so every
    existing knob — --subset, --tree_version, --llm, --max_beam_size, ... — still works)."""
    extra = argparse.ArgumentParser(add_help=False, description="General LATTICE retrieval")
    extra.add_argument("--query", type=str, default=None, help="A single query to retrieve for.")
    extra.add_argument("--queries_file", type=str, default=None, help="Path to a file with one query per line.")
    extra.add_argument("--top_k", type=int, default=10, help="How many leaf documents to return per query.")
    extra.add_argument("--tree_path", type=str, default=None,
                       help="Load this tree pickle directly instead of trees/{DATASET}/{SUBSET}/tree-{TREE_VERSION}.pkl.")
    extra.add_argument("--relevance_definition", type=str, default=None,
                       help="Custom definition of relevance injected into the traversal prompt (recommended for non-BRIGHT trees).")
    extra.add_argument("--output", type=str, default=None, help="Write results here as JSON (default: print to stdout).")
    extra.add_argument("--show_error_logs", action="store_true",
                       help="Log when a traversal response is repaired, falls back, or is dropped (useful for local/vLLM).")
    ns, remaining = extra.parse_known_args(argv)

    # HyperParams still requires --subset and --tree_version (they locate the default tree
    # and the per-subset relevance definition). Give friendly defaults so the CLI is usable
    # even when pointing at an arbitrary --tree_path.
    if "--subset" not in remaining:
        remaining += ["--subset", "general"]
    if "--tree_version" not in remaining:
        remaining += ["--tree_version", "custom"]
    hp = HyperParams.from_args(" ".join(remaining))
    return ns, hp


def _load_queries(ns) -> list[str]:
    if ns.query:
        return [ns.query]
    if ns.queries_file:
        with open(ns.queries_file, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    # Fall back to stdin (one query per line) so the CLI is pipe-friendly.
    if not sys.stdin.isatty():
        return [line.strip() for line in sys.stdin if line.strip()]
    raise SystemExit("No query provided. Use --query, --queries_file, or pipe queries on stdin.")


def main(argv=None):
    ns, hp = _parse_cli(argv)
    logger = _default_logger()
    logger.info("Hyperparams: %s", {k: v for k, v in vars(hp).items()})

    retriever = LatticeRetriever.from_hp(
        hp, logger=logger, tree_path=ns.tree_path, relevance_definition=ns.relevance_definition,
        show_error_logs=ns.show_error_logs,
    )
    queries = _load_queries(ns)
    logger.info("Retrieving for %d query/queries (num_iters=%d, top_k=%d)", len(queries), hp.NUM_ITERS, ns.top_k)

    all_results = retriever.retrieve_many(queries, num_iters=hp.NUM_ITERS, top_k=ns.top_k)
    payload = [{"query": q, "results": rows} for q, rows in zip(queries, all_results)]

    if ns.output:
        with open(ns.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote %d result set(s) to %s", len(payload), ns.output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
#endregion
