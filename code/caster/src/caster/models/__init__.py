from .base_adapter import BaseCandidateAdapter, validate_adapter_contract
from .registry import CandidateSpec, REGISTRY_COLUMNS, apply_hyperparam_overrides, hash_hyperparams, make_registry, validate_registry, read_registry, write_registry
from .toy_adapters import DriftAdapter, LastValueAdapter
from .mechanistic_adapters import SIRAdapter as MeanFieldSIRAdapter, SEIRAdapter as MeanFieldSEIRAdapter
from .renewal_adapters import RenewalAdapter, LocalLevelStateSpaceAdapter
from .neural_adapters import MLPAdapter, RNNAdapter
from .retrieval import (
    FORMAL_RETRIEVAL_PROFILE,
    RetrievalConfig,
    embed_registry,
    hashed_text_embedding,
    select_top_k_candidates,
    select_top_k_candidates_formal,
)
from .qwen25_embedding import (
    QWEN25_EMBEDDING_MANIFEST_SCHEMA,
    QWEN25_EMBEDDING_PROFILE,
    Qwen25EmbeddingBatch,
    Qwen25EmbeddingConfig,
    Qwen25HiddenStateEmbedder,
    embed_registry_qwen25,
    fingerprint_local_checkpoint,
)
from .selection_validation import build_candidate_validation_scores, build_test_rmse_ranking, task_macro_rmse
from .adapter_factory import instantiate_adapter_from_row, instantiate_adapters_from_registry
from .foundation_adapters import ExternalForecastAdapter, ChronosForecastAdapter, TimesFMForecastAdapter, TimeGPTForecastAdapter
__all__ = [
    "BaseCandidateAdapter", "validate_adapter_contract",
    "CandidateSpec", "REGISTRY_COLUMNS", "apply_hyperparam_overrides", "hash_hyperparams", "make_registry", "validate_registry", "read_registry", "write_registry",
    "DriftAdapter", "LastValueAdapter", "SIRAdapter", "SEIRAdapter", "MeanFieldSIRAdapter", "MeanFieldSEIRAdapter",
    "RenewalAdapter", "LocalLevelStateSpaceAdapter", "MLPAdapter", "RNNAdapter",
    "FORMAL_RETRIEVAL_PROFILE", "RetrievalConfig", "embed_registry", "hashed_text_embedding",
    "select_top_k_candidates", "select_top_k_candidates_formal",
    "QWEN25_EMBEDDING_MANIFEST_SCHEMA", "QWEN25_EMBEDDING_PROFILE",
    "Qwen25EmbeddingBatch", "Qwen25EmbeddingConfig", "Qwen25HiddenStateEmbedder",
    "embed_registry_qwen25", "fingerprint_local_checkpoint",
    "build_candidate_validation_scores", "build_test_rmse_ranking", "task_macro_rmse",
    "instantiate_adapter_from_row", "instantiate_adapters_from_registry",
    "ExternalForecastAdapter", "ChronosForecastAdapter", "TimesFMForecastAdapter", "TimeGPTForecastAdapter",
]

                                                 
from .candidate_adapters import (
    LastValueReferenceAdapter,
    SeasonalNaiveReferenceAdapter,
    DriftReferenceAdapter,
    CovariateDriftReferenceAdapter,
    SIRReferenceAdapter,
    SEIRReferenceAdapter,
    SEIRSReferenceAdapter,
    TimeVaryingSEIRAdapter,
    RenewalRtReferenceAdapter,
    LocalLevelDLMAdapter,
    _DampedTrendCoreAdapter,
    CovariateDynamicLinearTrendDLMAdapter,
    ParticleFilteredLocalLevelAdapter,
    AutoARIMAReferenceAdapter,
    AutoETSReferenceAdapter,
    AutoThetaReferenceAdapter,
    AutoCESReferenceAdapter,
    ProphetReferenceAdapter,
    RNNReferenceAdapter,
    LSTMReferenceAdapter,
    GRUReferenceAdapter,
    DeepARStyleAdapter,
    NBEATSReferenceAdapter,
    NHITSReferenceAdapter,
    PatchTSTReferenceAdapter,
    TFTReferenceAdapter,
)

SIRAdapter = SIRReferenceAdapter
SEIRAdapter = SEIRReferenceAdapter

__all__ += [
    "LastValueReferenceAdapter", "SeasonalNaiveReferenceAdapter", "DriftReferenceAdapter", "CovariateDriftReferenceAdapter",
    "SIRReferenceAdapter", "SEIRReferenceAdapter", "SEIRSReferenceAdapter", "TimeVaryingSEIRAdapter",
    "RenewalRtReferenceAdapter", "LocalLevelDLMAdapter", "_DampedTrendCoreAdapter", "CovariateDynamicLinearTrendDLMAdapter", "ParticleFilteredLocalLevelAdapter",
    "AutoARIMAReferenceAdapter", "AutoETSReferenceAdapter", "AutoThetaReferenceAdapter", "AutoCESReferenceAdapter", "ProphetReferenceAdapter",
    "RNNReferenceAdapter", "LSTMReferenceAdapter", "GRUReferenceAdapter", "DeepARStyleAdapter",
    "NBEATSReferenceAdapter", "NHITSReferenceAdapter", "PatchTSTReferenceAdapter", "TFTReferenceAdapter",
]
