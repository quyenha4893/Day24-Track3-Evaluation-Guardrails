"""A.1 — Generate 50 questions from corpus with distribution 50/25/25.

RAGAS 0.2.x API:
- TestsetGenerator.from_langchain(llm, embedding_model) wraps langchain objects internally
- Pass raw ChatOpenAI (not pre-wrapped LangchainLLMWrapper) to avoid double-wrapping
- Use RunConfig(max_workers=1) to prevent race conditions in LangchainLLMWrapper.agenerate_text
- query_distribution = [(synthesizer, weight), ...]
- Columns: user_input, reference_contexts, reference, synthesizer_name
"""

import sys
import os
# Force UTF-8 output on Windows
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.engine import Parallel, apply_transforms
from ragas.testset.transforms.extractors import EmbeddingExtractor, HeadlinesExtractor
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor, SummaryExtractor, ThemesExtractor,
)
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
from ragas.testset.transforms.relationship_builders.traditional import OverlapScoreBuilder
from ragas.testset.transforms.splitters import HeadlineSplitter

from config import RAG_DATA_DIR, PHASE_A_DIR, TEST_SET_SIZE, TEST_DISTRIBUTION, GENERATOR_MODEL


def build_custom_transforms(docs, llm, embedding_model):
    """
    Build custom transforms with lowered CosineSimilarityBuilder threshold
    to allow MultiHopAbstractQuerySynthesizer to find clusters even with
    different-topic documents (Vietnamese legal text + financial report).
    """
    def filter_doc_with_num_tokens(node, min_num_tokens=500):
        from ragas.utils import num_tokens_from_string
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties.get("page_content", "")) > min_num_tokens
        )

    def filter_chunks(node):
        return node.type == NodeType.CHUNK

    headline_extractor = HeadlinesExtractor(
        llm=llm,
        filter_nodes=lambda node: filter_doc_with_num_tokens(node),
    )
    splitter = HeadlineSplitter(min_tokens=500)
    summary_extractor = SummaryExtractor(
        llm=llm,
        filter_nodes=lambda node: filter_doc_with_num_tokens(node),
    )
    theme_extractor = ThemesExtractor(
        llm=llm,
        filter_nodes=lambda node: filter_chunks(node),
    )
    ner_extractor = NERExtractor(
        llm=llm,
        filter_nodes=lambda node: filter_chunks(node),
    )
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: filter_doc_with_num_tokens(node),
    )
    # Use very low threshold (0.01) to ensure clusters are created even for
    # topically different documents (legal + financial) - needed for MultiHopAbstract
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.01,  # Low threshold to create cross-document clusters
        filter_nodes=lambda node: filter_doc_with_num_tokens(node),
    )
    ner_overlap_sim = OverlapScoreBuilder(
        threshold=0.01,
        filter_nodes=lambda node: filter_chunks(node),
    )
    node_filter = CustomNodeFilter(
        llm=llm,
        filter_nodes=lambda node: filter_chunks(node),
    )

    return [
        headline_extractor,
        splitter,
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]


def main():
    print(f"Loading corpus from {RAG_DATA_DIR}")
    loader = DirectoryLoader(
        str(RAG_DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # RAGAS 0.2.x: from_langchain wraps ChatOpenAI internally.
    # Pass raw ChatOpenAI (not pre-wrapped LangchainLLMWrapper) to avoid double-wrapping:
    #   from_langchain does LangchainLLMWrapper(llm) internally.
    chat_llm = ChatOpenAI(model=GENERATOR_MODEL, temperature=0)
    oai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Wrap once for use in transforms and synthesizers (not passed to from_langchain)
    llm = LangchainLLMWrapper(chat_llm)
    embeddings = LangchainEmbeddingsWrapper(oai_embeddings)

    # from_langchain receives raw langchain objects (wraps them internally)
    generator = TestsetGenerator.from_langchain(
        llm=chat_llm,
        embedding_model=oai_embeddings,
    )

    # Build query_distribution: list of (synthesizer, weight) tuples
    # simple -> SingleHop (50%), reasoning -> MultiHopAbstract (25%), multi_context -> MultiHopSpecific (25%)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), TEST_DISTRIBUTION["simple"]),
        (MultiHopAbstractQuerySynthesizer(llm=llm), TEST_DISTRIBUTION["reasoning"]),
        (MultiHopSpecificQuerySynthesizer(llm=llm), TEST_DISTRIBUTION["multi_context"]),
    ]

    # Use max_workers=1 to avoid race conditions in LangchainLLMWrapper.agenerate_text
    # (concurrent coroutines mutate langchain_llm.temperature non-thread-safely)
    run_config = RunConfig(max_workers=1, timeout=600)

    print("Building knowledge graph from documents ...")
    nodes = [
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        for doc in docs
    ]
    kg = KnowledgeGraph(nodes=nodes)

    # Apply custom transforms with low CosineSimilarityBuilder threshold
    # so MultiHopAbstractQuerySynthesizer can find cross-document clusters
    transforms = build_custom_transforms(docs, llm, embeddings)
    apply_transforms(kg, transforms, run_config=run_config)

    # Verify KG is populated
    nodes_with_summary = [n for n in kg.nodes if n.properties.get("summary_embedding") is not None]
    print(f"Nodes with summary_embedding: {len(nodes_with_summary)} / {len(kg.nodes)}")
    rels_with_sim = [r for r in kg.relationships if r.get_property("summary_similarity")]
    print(f"Relationships with summary_similarity: {len(rels_with_sim)}")

    if len(nodes_with_summary) == 0:
        raise RuntimeError(
            "No nodes populated summaries — transforms all failed. Check API key and connectivity."
        )

    # Set the pre-built KG on the generator
    generator.knowledge_graph = kg

    print(f"Generating {TEST_SET_SIZE} questions ...")
    testset = generator.generate(
        testset_size=TEST_SET_SIZE,
        query_distribution=query_distribution,
        run_config=run_config,
        raise_exceptions=False,
        with_debugging_logs=False,
    )

    df = testset.to_pandas()
    print(f"Raw columns: {df.columns.tolist()}")
    print(f"Raw rows: {len(df)}")

    # Map RAGAS 0.2.x column names to expected format
    # user_input -> question, reference -> ground_truth,
    # reference_contexts -> contexts, synthesizer_name -> evolution_type
    col_map = {}
    if "user_input" in df.columns:
        col_map["user_input"] = "question"
    if "reference" in df.columns:
        col_map["reference"] = "ground_truth"
    if "reference_contexts" in df.columns:
        col_map["reference_contexts"] = "contexts"
    elif "retrieved_contexts" in df.columns:
        col_map["retrieved_contexts"] = "contexts"
    if "synthesizer_name" in df.columns:
        col_map["synthesizer_name"] = "evolution_type"
    df = df.rename(columns=col_map)

    # Ensure we have the required columns (add placeholders if missing)
    for col in ["question", "ground_truth", "contexts", "evolution_type"]:
        if col not in df.columns:
            df[col] = None

    assert len(df) >= 50, f"Need >=50 rows, got {len(df)}"
    required = {"question", "ground_truth", "contexts", "evolution_type"}
    assert required.issubset(df.columns), f"Missing cols: {required - set(df.columns)}"

    out = PHASE_A_DIR / "testset_v1.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")

    dist = df["evolution_type"].value_counts().to_dict()
    print(f"Distribution: {dist}")

    review_path = PHASE_A_DIR / "testset_review_notes.md"
    if not review_path.exists():
        review_path.write_text(
            "# Test Set Review Notes\n\n"
            "## Distribution check\n\n"
            f"```\n{dist}\n```\n\n"
            "## Manual review (>= 10 questions)\n\n"
            "| # | Question | Verdict | Note |\n"
            "|---|---|---|---|\n"
            "| 1 | ... | OK | ... |\n"
            "| 2 | ... | EDITED | reword for clarity |\n\n"
            "## Edits applied\n\n"
            "- Q#2: reword for clarity\n",
            encoding="utf-8",
        )
    print(f"Review template -> {review_path}")


if __name__ == "__main__":
    main()
