# Tutorial 6: Sparse Autoencoders

**Paper**: [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/) by Anthropic (2023)

**Difficulty**: Advanced | **Time**: 2-3 hours

**Status**: ✅ Complete

---

## Overview

This tutorial demonstrates how to train and analyze Sparse Autoencoders (SAEs) to decompose polysemantic neurons into interpretable monosemantic features. We'll reproduce key concepts from Anthropic's landmark paper on dictionary learning for language models.

## Learning Objectives

By the end of this tutorial, you will understand:

1. **The Problem**: Why individual neurons are polysemantic (respond to multiple unrelated concepts)
2. **The Cause**: How superposition enables models to represent more features than dimensions
3. **The Solution**: How sparse autoencoders decompose superposed representations
4. **The Practice**: How to train SAEs and interpret learned features using mlxterp

## Prerequisites

- Basic understanding of neural networks and transformers
- Familiarity with mlxterp's tracing API
- Python programming experience
- ~8GB RAM for running examples

## Quick Start

```bash
# Run the interactive tutorial
python examples/tutorials/06_sparse_autoencoders/sae_tutorial.py
```

---

## Part 1: The Polysemanticity Problem

### What is Polysemanticity?

In neural networks, individual neurons often respond to multiple, seemingly unrelated concepts. For example, a single neuron in a language model might activate for:

- The word "Paris" (the city)
- The word "France" (the country)
- Images of the Eiffel Tower
- French language text
- The number 1889 (year Eiffel Tower was built)

This makes interpretation difficult! We want **monosemantic features**: neurons that represent single, interpretable concepts.

### The Cause: Superposition

Models learn to represent MORE features than they have dimensions by using **superposition** - encoding multiple features in overlapping neural directions. This is efficient but makes individual neurons polysemantic.

**Intuition**: Imagine trying to store 1000 important concepts in 100 neurons. The model learns to pack them by using combinations and directions - but this makes each neuron respond to multiple concepts.

### Why This Matters for Interpretability

If we can't understand what individual neurons represent, we can't:
- Predict model behavior reliably
- Debug failures systematically
- Ensure safety properties
- Build trust in model decisions

**The goal**: Decompose polysemantic neurons into monosemantic features.

---

## Part 2: Sparse Autoencoders - The Solution

### Core Idea

Sparse Autoencoders (SAEs) learn an **overcomplete dictionary** of features:

```
Original Activation: 2048 dimensions (polysemantic neurons)
                ↓
SAE Encode: 65,536 features (32x overcomplete)
            Only ~64 features active (sparse!)
            Each feature is monosemantic
                ↓
SAE Decode: 2048 dimensions (reconstructed)
```

### Architecture

```python
class SAE:
    def encode(activation):
        # Project to overcomplete space + apply sparsity
        features = TopK(W_enc @ activation + b_enc, k=64)
        return features  # Most features are zero!

    def decode(features):
        # Project back to original space
        reconstruction = W_dec @ features + b_dec
        return reconstruction
```

**Key properties:**
- **Overcomplete**: More features (65k) than input dims (2k)
- **Sparse**: Only ~64 features active at once (99% are zero)
- **Reconstruction**: Decode(Encode(x)) ≈ x

### Training Objective

```python
loss = reconstruction_loss + sparsity_penalty

# Reconstruction: How well can we reconstruct original activations?
reconstruction_loss = MSE(original, reconstructed)

# Sparsity: Encourage few features to be active
# (TopK activation handles this automatically)
```

---

## Part 3: Training an SAE with mlxterp

mlxterp has built-in SAE support! Here's how to train one:

```python
from mlx_lm import load
from mlxterp import InterpretableModel, SAEConfig

# Load model
base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Prepare training data
dataset = [
    "Paris is the capital of France",
    "Python is a programming language",
    "Water freezes at zero degrees Celsius",
    # ... add more diverse examples
]

# Configure SAE
config = SAEConfig(
    sae_type="batchtopk",      # Modern architecture
    expansion_factor=32,        # 32x overcomplete
    k=64,                       # Top-64 sparsity
    learning_rate=3e-4,
    num_epochs=5,
    use_ghost_grads=True,       # Reduce dead features
)

# Train SAE on layer 10 MLP
sae = model.train_sae(
    layer=10,
    component="mlp",
    dataset=dataset,
    config=config,
    save_path="my_sae.mlx"
)
```

### Training Tips

**Which layer to use?**
- **Early layers (0-5)**: Simple features (tokens, syntax)
- **Middle layers (6-12)**: Semantic features (concepts, entities)
- **Late layers (13+)**: Task-specific features (predictions, reasoning)

**How much data?**
- Minimum: ~100 diverse examples
- Good: ~1000 examples
- Ideal: ~10,000+ examples (like the paper)

**Expansion factor:**
- Higher = more features, more monosemantic
- But also more dead features (features that never activate)
- Paper uses 8x to 512x, we recommend 16x to 32x

---

## Part 4: Analyzing Learned Features

### Finding Active Features

```python
# Get activations for a text
with model.trace("Paris is the capital of France") as trace:
    mlp_out = trace.activations['model.model.layers.10.mlp']

# Encode to features
features = sae.encode(mlp_out)

# Find top active features at last token
last_token_features = features[0, -1, :]
top_indices = mx.argsort(-mx.abs(last_token_features))[:10]

print("Top 10 active features:")
for idx in top_indices:
    activation = last_token_features[int(idx)].item()
    print(f"  Feature {int(idx):5d}: {activation:7.3f}")
```

### Interpreting Features

To understand what a feature represents:

1. **Collect activation examples**: Find inputs where it activates strongly
2. **Look for patterns**: What do these inputs have in common?
3. **Form hypothesis**: What concept does this feature encode?
4. **Test hypothesis**: Does it activate for new examples of this concept?

**Example Analysis:**

```python
# Test feature 1234 on various inputs
test_examples = [
    "Paris is the capital of France",      # Geographic
    "London is the capital of England",    # Geographic
    "Tokyo is in Japan",                   # Geographic
    "Python is a language",                # Non-geographic
    "def function(): pass",                # Code
]

# Hypothesis: Feature 1234 encodes "capital cities"
for text in test_examples:
    # ... compute activation ...
    # Expect: High for first 3, low for last 2
```

### Reconstruction Quality

Check how well the SAE reconstructs activations:

```python
# Original activation
with model.trace(text) as trace:
    original = trace.activations['model.model.layers.10.mlp']

# Reconstruct via SAE
features = sae.encode(original)
reconstructed = sae.decode(features)

# Measure quality
mse = mx.mean((original - reconstructed) ** 2).item()
relative_error = (mse / mx.mean(original ** 2).item()) * 100

print(f"Reconstruction MSE: {mse:.6f}")
print(f"Relative error: {relative_error:.2f}%")
```

**Good reconstruction**: < 5% relative error
**Acceptable**: 5-15% relative error
**Poor**: > 15% relative error (need more training or better config)

---

## Part 5: Feature Properties

### Sparsity

SAEs learn sparse features - most are zero at any given time:

```python
features = sae.encode(activation)
active_features = mx.sum(features != 0, axis=-1)
sparsity = (1 - active_features / sae.feature_dim) * 100

print(f"Active features: {active_features.item():.0f} / {sae.feature_dim}")
print(f"Sparsity: {sparsity.item():.1f}%")
# Typical: 64 active / 65,536 total = 99.9% sparse
```

### Dead Features

Some features never activate during training - these are "dead":

```python
# Track during training (mlxterp does this automatically)
dead_features = sae.get_dead_features()
print(f"Dead features: {len(dead_features)} / {sae.feature_dim}")
print(f"Dead feature rate: {len(dead_features) / sae.feature_dim * 100:.1f}%")
```

**Ghost gradients** (enabled in config) help reduce dead features:
- Without: ~95% dead features
- With: ~70% dead features

---

## Part 6: Comparison with Paper Results

### Key Findings from Anthropic's Paper

1. **Monosemanticity**: SAE features are significantly more monosemantic than neurons
2. **Interpretability**: Features correspond to clear concepts (programming, cities, etc.)
3. **Universality**: Similar features appear across different models
4. **Compositionality**: Complex behaviors emerge from feature combinations

### What We Demonstrated

✅ **Training**: Successfully trained SAE on model activations
✅ **Sparsity**: Achieved high sparsity (~99%) with TopK activation
✅ **Reconstruction**: Good reconstruction quality (< 10% error typical)
✅ **Feature Analysis**: Identified which features activate for different inputs
✅ **Patterns**: Found features that respond to semantic concepts

### Differences from Paper

| Aspect | Paper (Anthropic) | This Tutorial |
|--------|------------------|---------------|
| Model | Claude (large) | Llama-3.2-1B (small) |
| Training Data | Millions of examples | 20-100 examples |
| Expansion | Up to 512x | 32x |
| Compute | Large-scale GPUs | Single Mac (Apple Silicon) |
| Features | Highly monosemantic | Moderately monosemantic |

**Key insight**: The technique works at any scale, but larger models + more data = clearer features.

---

## Part 7: Advanced Topics

### Feature Steering (Conceptual)

Once we have interpretable features, we can steer model behavior:

```python
# Conceptual API (future implementation)
# Ablate a feature (remove its influence)
with model.trace("Text") as trace:
    model.ablate_sae_feature(
        sae=sae,
        layer=10,
        feature_id=1234,  # Remove "Paris" feature
    )
    output = trace.activations['__model_output__']

# Amplify a feature (strengthen its influence)
with model.trace("Text") as trace:
    model.amplify_sae_feature(
        sae=sae,
        layer=10,
        feature_id=5678,  # Amplify "programming" feature
        strength=2.0,
    )
```

This is more precise than traditional steering vectors because we're operating on interpretable feature directions!

### Multi-Layer SAEs

Train SAEs on different layers to understand feature hierarchy:

```python
# Early layer: simple features
sae_layer3 = model.train_sae(layer=3, dataset=dataset, ...)

# Middle layer: semantic features
sae_layer10 = model.train_sae(layer=10, dataset=dataset, ...)

# Late layer: abstract features
sae_layer15 = model.train_sae(layer=15, dataset=dataset, ...)
```

### Feature Visualization

Build dashboards to explore features interactively:
- See maximally activating examples
- Compare features across layers
- Track feature evolution during generation
- Visualize feature co-occurrence patterns

---

## Exercises

### Exercise 1: Train Your Own SAE

Train an SAE on a different layer and component:

```python
# Try layer 5 attention instead of layer 10 MLP
sae_attn = model.train_sae(
    layer=5,
    component="self_attn",  # Different component!
    dataset=dataset,
    config=config,
)
```

**Questions:**
- How do attention features differ from MLP features?
- Which layer has more interpretable features?

### Exercise 2: Feature Interpretation

Pick a feature and interpret it systematically:

1. Find 20 examples where it activates strongly
2. Find 20 examples where it doesn't activate
3. Form a hypothesis about what it represents
4. Test with new examples

### Exercise 3: Expansion Factor Study

Train SAEs with different expansion factors:

```python
for expansion in [8, 16, 32, 64]:
    config = SAEConfig(expansion_factor=expansion, ...)
    sae = model.train_sae(...)
    # Compare reconstruction quality and sparsity
```

**Questions:**
- How does expansion affect reconstruction quality?
- How does it affect dead feature rate?
- What's the optimal trade-off?

### Exercise 4: Domain-Specific Features

Train SAEs on domain-specific data:

```python
# Code-focused dataset
code_dataset = [
    "def fibonacci(n): return n if n < 2 else",
    "class MyClass: def __init__(self):",
    "import numpy as np",
    # ... more code examples
]

sae_code = model.train_sae(layer=10, dataset=code_dataset, ...)

# Do features specialize to code concepts?
```

---

## Limitations and Future Directions

### Current Limitations

1. **Dead Features**: Even with ghost gradients, ~70% of features are dead
2. **Polysemanticity**: Some features still respond to multiple concepts
3. **Computational Cost**: Large expansion factors require significant memory
4. **Interpretation**: Manual feature interpretation is time-consuming

### Future Directions

From the paper and ongoing research:

1. **Automated Interpretation**: Use LLMs to automatically label features
2. **Feature Circuits**: Track how features compose across layers
3. **Causal Features**: Identify features that causally affect outputs
4. **Cross-Model Features**: Find universal features across models
5. **Steering Applications**: Use features for precise behavior control

### mlxterp Roadmap

See `SAE_ROADMAP.md` for planned features:
- ✅ Phase 1: Training infrastructure (complete)
- 🔄 Phase 2: Feature analysis tools (in progress)
- 🔄 Phase 3: Advanced architectures (future)
- 🔄 Phase 4: Circuit discovery (future)

---

## Summary

### Key Takeaways

1. **Problem**: Individual neurons are polysemantic due to superposition
2. **Solution**: Sparse autoencoders decompose activations into monosemantic features
3. **Method**: Train overcomplete SAEs with sparsity constraints
4. **Result**: Interpretable features that represent clear concepts
5. **Application**: Use features for analysis and steering

### What We Built

✅ Trained a 32x overcomplete SAE on Layer 10 MLP
✅ Analyzed which features activate for different inputs
✅ Measured reconstruction quality and sparsity
✅ Interpreted feature meanings through activation patterns
✅ Understood the monosemanticity decomposition

### Next Steps

- Read the [full Anthropic paper](https://transformer-circuits.pub/2023/monosemantic-features/)
- Explore `examples/sae_realistic_training.py` for advanced training
- Check `SAE_ROADMAP.md` for upcoming features
- Try training SAEs on your own datasets
- Join the mechanistic interpretability community!

---

## References

### Primary Paper

1. **Towards Monosemanticity** (Anthropic, 2023)
   - [Paper](https://transformer-circuits.pub/2023/monosemantic-features/)
   - Demonstrates SAEs can decompose polysemantic neurons into interpretable features
   - Shows features are universal across models

### Related Work

2. **Sparse Autoencoders Find Highly Interpretable Features** (Cunningham et al., 2023)
   - [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
   - Theoretical foundations for SAE interpretability

3. **Toy Models of Superposition** (Elhage et al., 2022)
   - [Anthropic](https://transformer-circuits.pub/2022/toy_model/index.html)
   - Explains why superposition occurs

4. **Scaling Monosemanticity** (Anthropic, 2024)
   - [Paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
   - Scales SAEs to larger models with millions of features

### Additional Resources

- [SAELens](https://github.com/jbloomAus/SAELens): PyTorch library for SAE training
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens): Interpretability tools
- [Anthropic's Transformer Circuits Thread](https://transformer-circuits.pub/): Research on mech interp

### mlxterp Documentation

- [SAE Roadmap](../../SAE_ROADMAP.md): Development phases
- [Examples](../../examples/): More SAE examples
- [API Reference](../API.md): Complete API documentation
