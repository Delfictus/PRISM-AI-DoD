pub mod gpu_llm_inference;
pub mod gpu_transformer;
pub mod gguf_loader;
pub mod gguf_gpu_loader;
pub mod kv_cache;
pub mod bpe_tokenizer;
pub mod sampling;
pub mod llm_metrics;
pub mod attention_analyzer;
pub mod transfer_entropy_llm;
pub mod speculative_decoding;
pub mod llm_analysis;
pub mod phase6_adapters;
pub mod phase6_llm_adapters;
pub mod gpu_neural_enhancements;
pub mod gpu_visual_embeddings;
pub mod gpu_inference_optimization;
pub mod gpu_protein_folding;
pub mod gpu_deep_graph_protein;
pub mod gpu_protein_training;
pub mod gpu_cutlass_kernels;
pub mod pdb_dataset;

pub use gpu_llm_inference::{
    GpuLocalLLMSystem,
    LLMArchitecture,
    ModelConfig,
    SimpleTokenizer,
};

pub use gpu_transformer::{
    GpuTransformerLayer,
    GpuLLMInference,
};

pub use gguf_loader::{
    GgufLoader,
    GgufType,
    MetadataValue,
    TensorInfo,
};

pub use gguf_gpu_loader::{
    GgufGpuLoader,
};

pub use kv_cache::{
    LayerKVCache,
    TransformerKVCache,
    KVCacheStats,
};

pub use bpe_tokenizer::{
    BPETokenizer,
    SpecialTokens,
};

pub use sampling::{
    TokenSampler,
    SamplingConfig,
};

pub use llm_metrics::{
    LLMMetrics,
    DistributionHealth,
};

pub use attention_analyzer::{
    AttentionAnalyzer,
    AttentionHealth,
    AttentionStats,
};

pub use transfer_entropy_llm::{
    TransferEntropyLLM,
    TransferEntropyStats,
};

pub use speculative_decoding::{
    SpeculativeDecoder,
    SelfSpeculativeDecoder,
    SpeculativeStats,
};

pub use llm_analysis::{
    LLMAnalysis,
};

pub use phase6_adapters::{
    // Trait interfaces for Phase 6 enhancements
    GnnConsensusAdapter,
    TdaTopologyAdapter,
    MetaLearningAdapter,

    // Supporting types
    TopologyFeatures,
    GenerationContext,
    AnalysisStrategy,
    SamplingParams,
    QueryType,
    PerformanceRequirements,
    DecodingStrategy,

    // Placeholder implementations (until Phase 6)
    PlaceholderGnnAdapter,
    PlaceholderTdaAdapter,
    PlaceholderMetaLearningAdapter,
};

pub use phase6_llm_adapters::{
    // LLM-specific Phase 6 implementations
    LlmTdaAdapter,
    LlmMetaLearningAdapter,
    LlmGnnAdapter,
};

pub use gpu_neural_enhancements::{
    // GPU-accelerated neural enhancements
    GpuCnnAttentionProcessor,
    GpuEmbeddingTransformer,
    MultiModalFusionProcessor,
    GpuAttentionAnalyzer,
    AttentionVisualFeatures,
    AttentionPattern,
    ComprehensiveAttentionAnalysis,

    // Protein folding prediction types
    ProteinStructureFeatures,
    SecondaryStructure,
    ContactRanges,
};

pub use gpu_visual_embeddings::{
    // CNN-style visual processing
    GpuResNetVisual,
    VisionTransformerPatches,
    VisualTextAligner,
    AttentionToImageConverter,
};

pub use gpu_inference_optimization::{
    // GPU-accelerated inference optimization
    FlashAttention,
    DynamicQuantizer,
    KVCacheCompressor,
    TokenVisualGrounding,
    QuantizedTensor,
    QuantizationParams,
    CompressedKVCache,
    GroundingResult,
    BoundingBox,
    MemoryStats,
    CompressionStats,
};

pub use gpu_protein_folding::{
    // GPU-accelerated neuromorphic-topological protein folding
    GpuProteinFoldingSystem,
    ProteinPrediction,
    FreeEnergyAnalysis,
    EntropyAnalysis,
    BindingPocket,
    PhaseDynamicsAnalysis,
    FoldingDynamics,
    EnergyWeights,
};

pub use gpu_deep_graph_protein::{
    // Deep multi-scale graph neural network for ultra-accurate protein folding
    DeepGraphProteinFolder,
    DeepProteinPrediction,
    DeepGraphConfig,
    AccuracyMetrics,
};

pub use gpu_protein_training::{
    // Full GPU acceleration + training capability for protein folding
    FullGpuProteinSystem,
    TrainingConfig,
    TrainingMetrics,
    LossFunction,
};

pub use gpu_cutlass_kernels::{
    // CUTLASS 3.8 + FlashAttention-3 for cutting-edge GPU acceleration
    CutlassKernels,
    TensorOps,
    TensorCoreArch,
    ReductionOp,
    ElementwiseOp,
};

pub use pdb_dataset::{
    // PDB dataset loading for supervised training
    ProteinDataset,
    SecondaryStructure,
    ProteinMetadata,
    Batch,
};
