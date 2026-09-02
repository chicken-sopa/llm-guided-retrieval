"""LATTICE: LLM-guided retrieval by traversing a semantic tree.

This package is the inference engine on its own -- no evaluation harness, no gold
labels, no metrics, no wandb -- so it can be installed into another project and
called directly:

    from lattice_core import HyperParams, LatticeRetriever

    hp = HyperParams.from_args("--dataset ECHR --subset convention "
                              "--tree_version bottom-up-mc10 "
                              "--llm_api_backend openai --llm gpt-4.1")
    retriever = LatticeRetriever.from_hp(hp)

    for hit in retriever.retrieve("Does Article 8 cover phone tapping?", top_k=10):
        print(hit["rank"], round(hit["score"], 3), hit["node_id"], hit["text"][:100])

Two things to know when embedding this in another system:

* From async code (an agent framework, a web service), await `retrieve_async` /
  `retrieve_many_async`. The sync `retrieve` wrappers give every call its own
  event loop, which an async host does not want.
* `HyperParams.from_args()` with no argument parses `sys.argv`, so inside another
  application ALWAYS pass an explicit string, as above.

Trees are loaded from `corpora/{DATASET}/{SUBSET}/trees/tree-{TREE_VERSION}.pkl`
relative to the repo root; pass `tree_path=` to `from_hp` to load one from anywhere.
"""

from ._async import run_coro_sync
from .hyperparams import HyperParams
from .lattice import (
    LatticeRetriever,
    TracingLatticeRetriever,
    build_llm_api_and_kwargs,
    load_semantic_tree,
    override_relevance_definition,
)
from .tree_objects import InferSample, SemanticNode

__all__ = [
    "HyperParams",
    "InferSample",
    "LatticeRetriever",
    "SemanticNode",
    "TracingLatticeRetriever",
    "build_llm_api_and_kwargs",
    "load_semantic_tree",
    "override_relevance_definition",
    "run_coro_sync",
]
