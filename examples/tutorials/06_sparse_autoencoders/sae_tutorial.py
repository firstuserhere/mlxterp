"""
Tutorial 6: Sparse Autoencoders for Decomposing Model Representations

This tutorial reproduces key findings from:
"Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
by Anthropic (2023)

Paper: https://transformer-circuits.pub/2023/monosemantic-features/

Key Concepts:
1. Polysemanticity and superposition
2. Sparse autoencoders for feature decomposition
3. Feature interpretation and analysis
4. Feature steering for controlled generation

Prerequisites:
- mlxterp installed with all dependencies
- ~8GB RAM for small models (Llama-3.2-1B)
- Apple Silicon Mac recommended
"""

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel, SAEConfig
from typing import List, Tuple
import numpy as np


def section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# ==============================================================================
# PART 1: Understanding Polysemanticity and Superposition
# ==============================================================================

section("PART 1: The Polysemanticity Problem")

print("""
THE PROBLEM: Polysemantic Neurons

In neural networks, individual neurons often respond to multiple, seemingly
unrelated concepts. For example, a single neuron might activate for:
  - The word "Paris" (the city)
  - The word "France" (the country)
  - Images of the Eiffel Tower
  - French language text
  - The number 1889 (year Eiffel Tower was built)

This makes interpretation difficult! We want monosemantic features: neurons
that represent single, interpretable concepts.

THE CAUSE: Superposition

Models learn to represent MORE features than they have dimensions by using
superposition - encoding multiple features in overlapping neural directions.
This is efficient but makes individual neurons polysemantic.

THE SOLUTION: Sparse Autoencoders (SAEs)

SAEs learn an overcomplete dictionary of features - more features than the
original dimension - where each feature is sparse (mostly zero). This
decomposes superposed representations into interpretable monosemantic features.

Key Insight:
    Original: 2048-dim activation (each neuron = polysemantic)
    SAE Output: 65,536 features (32x overcomplete, each feature = monosemantic)
                Only ~64-128 features active at once (sparse!)
""")

input("Press Enter to continue to training...")


# ==============================================================================
# PART 2: Training a Sparse Autoencoder
# ==============================================================================

section("PART 2: Training a Sparse Autoencoder")

print("Loading model...")
base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"Model has {len(model.layers)} layers")
print(f"Hidden dimension: {base_model.model.embed_tokens.weight.shape[1]}")

# Prepare diverse training data
print("\nPreparing training dataset...")
dataset = [
    # Geography and locations
    "Paris is the capital of France",
    "Tokyo is a major city in Japan",
    "The Amazon rainforest is in South America",

    # Science and nature
    "Water freezes at zero degrees Celsius",
    "Photosynthesis converts sunlight into energy",
    "DNA contains genetic information",
    "The Earth orbits the Sun",

    # Technology and programming
    "Python is a programming language",
    "Machine learning uses neural networks",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Artificial intelligence is transforming technology",

    # History and culture
    "Shakespeare wrote Hamlet in the early 1600s",
    "The Roman Empire fell in 476 AD",
    "The Renaissance began in Italy",

    # Mathematics
    "The Pythagorean theorem: a² + b² = c²",
    "Pi is approximately 3.14159",
    "Prime numbers are divisible only by 1 and themselves",

    # Language and communication
    "Hello, how are you doing today?",
    "The quick brown fox jumps over the lazy dog",
    "Communication is essential for collaboration",
]

print(f"Training dataset: {len(dataset)} examples")

# Configure SAE with recommended settings from the paper
print("\nConfiguring SAE...")
config = SAEConfig(
    # Architecture
    sae_type="batchtopk",      # BatchTopK (modern, more stable)
    expansion_factor=32,        # 32x overcomplete (paper uses 8-512x)
    k=64,                       # Top-k sparsity (paper uses 32-512)

    # Training
    learning_rate=3e-4,         # Paper uses 1e-4 to 1e-3
    num_epochs=5,               # Keep it short for tutorial
    batch_size=8,

    # Advanced features
    use_ghost_grads=True,       # Reduces dead features
    lr_scheduler="cosine",      # Smooth learning rate decay

    # Monitoring
    use_wandb=False,            # Set to True to track in W&B
)

print(f"""
SAE Configuration:
  - Type: {config.sae_type}
  - Expansion: {config.expansion_factor}x
  - Sparsity: Top-{config.k} features
  - Learning rate: {config.learning_rate}
  - Epochs: {config.num_epochs}
  - Ghost gradients: {config.use_ghost_grads}
""")

# Train SAE on layer 10 MLP (middle layer typically has rich features)
print("\n🎯 Training SAE on Layer 10 MLP...")
print("This will take a few minutes...\n")

sae = model.train_sae(
    layer=10,
    component="mlp",
    dataset=dataset,
    config=config,
    save_path="tutorial_sae_layer10.mlx"
)

print("\n✅ SAE training complete!")
print(f"Trained {sae.expansion_factor}x overcomplete SAE")
print(f"Input dim: {sae.input_dim}, Feature dim: {sae.feature_dim}")

input("\nPress Enter to continue to feature analysis...")


# ==============================================================================
# PART 3: Analyzing Learned Features
# ==============================================================================

section("PART 3: Feature Analysis - What Did the SAE Learn?")

print("""
Now we'll analyze what features the SAE learned by:
1. Finding which features activate for specific inputs
2. Interpreting what concepts each feature represents
3. Finding maximally activating examples for features
""")

# Test 1: Geographic features
print("\n--- Test 1: Geographic Concepts ---")
test_texts = [
    "Paris is the capital of France",
    "Tokyo is a major city in Japan",
    "The Amazon rainforest is located in Brazil",
]

for text in test_texts:
    print(f"\nText: '{text}'")

    # Get activations
    with model.trace(text) as trace:
        mlp_out = trace.activations[f'model.model.layers.10.mlp']

    # Encode with SAE to get features
    features = sae.encode(mlp_out)

    # Find top activating features (at last token position)
    last_token_features = features[0, -1, :]
    top_k = 5
    top_indices = mx.argsort(-mx.abs(last_token_features))[:top_k]

    print(f"  Top {top_k} active features:")
    for idx in top_indices:
        activation = last_token_features[int(idx)].item()
        if abs(activation) > 0.01:  # Only show significant activations
            print(f"    Feature {int(idx):5d}: {activation:7.3f}")

# Test 2: Programming concepts
print("\n\n--- Test 2: Programming Concepts ---")
code_texts = [
    "def fibonacci(n): return n",
    "Python is a programming language",
    "class MyClass: pass",
]

for text in code_texts:
    print(f"\nText: '{text}'")

    with model.trace(text) as trace:
        mlp_out = trace.activations[f'model.model.layers.10.mlp']

    features = sae.encode(mlp_out)
    last_token_features = features[0, -1, :]
    top_indices = mx.argsort(-mx.abs(last_token_features))[:5]

    print(f"  Top 5 active features:")
    for idx in top_indices:
        activation = last_token_features[int(idx)].item()
        if abs(activation) > 0.01:
            print(f"    Feature {int(idx):5d}: {activation:7.3f}")

# Test 3: Reconstruction quality
print("\n\n--- Test 3: Reconstruction Quality ---")
print("How well does the SAE reconstruct original activations?")

test_text = "The capital of France is Paris"
with model.trace(test_text) as trace:
    original = trace.activations[f'model.model.layers.10.mlp']

# Encode and decode
features = sae.encode(original)
reconstructed = sae.decode(features)

# Compute reconstruction error
mse = mx.mean((original - reconstructed) ** 2).item()
relative_error = (mse / mx.mean(original ** 2).item()) * 100

print(f"\nReconstruction MSE: {mse:.6f}")
print(f"Relative error: {relative_error:.2f}%")

# Check sparsity
active_features = mx.sum(features != 0, axis=-1)
print(f"Active features: {active_features[0, -1].item():.0f} / {sae.feature_dim}")
print(f"Sparsity: {(1 - active_features[0, -1].item() / sae.feature_dim) * 100:.1f}%")

input("\nPress Enter to continue to feature interpretation...")


# ==============================================================================
# PART 4: Feature Interpretation
# ==============================================================================

section("PART 4: Interpreting Specific Features")

print("""
The key question in interpretability: What does each feature represent?

To interpret a feature, we:
1. Find examples where it activates strongly
2. Look for common patterns
3. Form a hypothesis about its meaning
4. Test with new examples

Let's analyze a few features that showed up in our tests.
""")

def analyze_feature(feature_id: int, test_examples: List[str]):
    """Analyze what a specific feature responds to."""
    print(f"\n{'─'*80}")
    print(f"Feature {feature_id} Analysis")
    print(f"{'─'*80}")

    activations = []
    for text in test_examples:
        with model.trace(text) as trace:
            mlp_out = trace.activations[f'model.model.layers.10.mlp']
        features = sae.encode(mlp_out)
        activation = features[0, -1, feature_id].item()
        activations.append((text, activation))

    # Sort by activation strength
    activations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nActivation strengths:")
    for text, activation in activations:
        bar = '█' * int(abs(activation) * 10)
        print(f"  {activation:7.3f} {bar:20s} | {text[:50]}")

# Example: Analyze a feature that activated for geographic text
print("\nLet's analyze a feature that might encode geographic information...")
geographic_examples = [
    "Paris is the capital of France",
    "London is the capital of England",
    "Tokyo is in Japan",
    "Mathematics is the study of numbers",  # Non-geographic control
    "def function(): pass",  # Code control
    "The Earth orbits the Sun",  # Science
]

# Pick the first feature that activated for Paris
with model.trace("Paris is the capital") as trace:
    mlp_out = trace.activations[f'model.model.layers.10.mlp']
features = sae.encode(mlp_out)
top_feature = int(mx.argmax(mx.abs(features[0, -1, :])).item())

analyze_feature(top_feature, geographic_examples)

input("\nPress Enter to continue to feature steering...")


# ==============================================================================
# PART 5: Feature Steering
# ==============================================================================

section("PART 5: Steering Model Behavior with Features")

print("""
Feature steering: Use SAE features to control model behavior!

By identifying monosemantic features, we can:
1. Ablate features to remove concepts
2. Amplify features to strengthen concepts
3. Combine features for complex steering

This is more targeted than traditional steering vectors because
we're operating on interpretable feature directions.

Note: Full feature steering requires additional implementation
(intervention system integration). This shows the concept.
""")

print("\n--- Demonstrating Feature Decomposition ---")

test_prompt = "The capital of France is"
print(f"\nPrompt: '{test_prompt}'")

# Get normal activations
with model.trace(test_prompt) as trace:
    mlp_out = trace.activations[f'model.model.layers.10.mlp']
    logits = trace.activations['__model_output__']

# Show what features are active
features = sae.encode(mlp_out)
last_features = features[0, -1, :]
top_k = 10
top_indices = mx.argsort(-mx.abs(last_features))[:top_k]

print(f"\nTop {top_k} active features for this prompt:")
for i, idx in enumerate(top_indices, 1):
    activation = last_features[int(idx)].item()
    print(f"  {i:2d}. Feature {int(idx):5d}: {activation:7.3f}")

# Get model predictions
predictions = model.get_token_predictions(logits[0, -1, :], top_k=5)
print(f"\nModel's top predictions:")
for token_id in predictions:
    print(f"  - '{model.token_to_str(token_id)}'")

print("""
Future work: Feature-level interventions
- model.ablate_sae_feature(sae, layer=10, feature_id=X)
- model.amplify_sae_feature(sae, layer=10, feature_id=X, strength=2.0)
- Combine multiple feature modifications for complex steering
""")

input("\nPress Enter to see summary...")


# ==============================================================================
# Summary and Key Takeaways
# ==============================================================================

section("Summary: Sparse Autoencoders for Interpretability")

print("""
KEY INSIGHTS FROM THIS TUTORIAL:

1. THE PROBLEM: Polysemanticity
   - Individual neurons respond to multiple unrelated concepts
   - This is caused by superposition (efficient but hard to interpret)

2. THE SOLUTION: Sparse Autoencoders
   - Learn an overcomplete dictionary of sparse features
   - 32x expansion: 2048 dims → 65,536 features
   - Only ~64 features active at once (sparse!)
   - Each feature is more monosemantic (single concept)

3. WHAT WE DEMONSTRATED:
   ✓ Trained an SAE on Layer 10 MLP activations
   ✓ Analyzed which features activate for different inputs
   ✓ Interpreted feature meanings by finding activation patterns
   ✓ Showed reconstruction quality and sparsity metrics

4. NEXT STEPS:
   - Train SAEs on different layers and components
   - Collect larger datasets for better feature learning
   - Implement feature steering interventions
   - Build feature dashboards for exploration
   - Compare features across different models

5. PAPER REPRODUCTION:
   This tutorial demonstrated core concepts from:
   "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
   https://transformer-circuits.pub/2023/monosemantic-features/

   The paper shows similar patterns at scale:
   - Larger expansion factors (up to 512x)
   - More training data (millions of examples)
   - Sophisticated feature interpretation
   - Feature universality across models

RESOURCES:
- Original paper: https://transformer-circuits.pub/2023/monosemantic-features/
- mlxterp SAE docs: See examples/sae_realistic_training.py
- Related: Cunningham et al. (2023) on interpretable features in LMs
""")

print("\n✅ Tutorial complete!")
print("\nTry experimenting with:")
print("  - Different layers (early layers: simple features, late layers: complex)")
print("  - Different expansion factors (higher = more features, but more dead features)")
print("  - Different sparsity levels (k parameter)")
print("  - Larger datasets for better feature learning")
print("\nSaved SAE to: tutorial_sae_layer10.mlx")
