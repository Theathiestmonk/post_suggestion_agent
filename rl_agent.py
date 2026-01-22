# rl_agent.py
import math
import random
import numpy as np
import db
from collections import defaultdict

# ---------------- ACTION SPACE ----------------

ACTION_SPACE = {

    "HOOK_TYPE": [
        "question",
        "bold_claim",
        "curiosity_gap",
        "relatable_pain",
        "problem_solution",
        "before_after",
        "transformation",
        "surprising_fact",
        "social_proof",
        "authority_expert",
        "aspirational_vision",
        "emotional_moment",
        "pattern_interrupt",
        "visual_metaphor",
        "contrast_comparison",
        "minimal_message",
        "trend_reference",
        "practical_tip",
        "myth_busting",
        "counter_intuitive_take"
    ],

    "INFORMATION_DEPTH": [
        "one_liner",
        "snackable",
        "balanced",
        "value_dense",
        "deep_dive",
        "story_arc",
        "visual_dominant"
    ],

    "TONE": [
        "calm",
        "confident",
        "professional",
        "friendly",
        "playful",
        "serious",
        "educational",
        "authoritative",
        "empathetic",
        "inspirational",
        "motivational",
        "premium",
        "bold",
        "warm",
        "cool",
        "modern",
        "timeless",
        "aspirational",
        "rebellious",
        "trust_reassuring"
    ],

    "CREATIVITY": [
        "ultra_safe",
        "safe",
        "balanced",
        "bold",
        "experimental",
        "highly_experimental"
    ],

    "VISUAL_STYLE": [
        "minimal_clean_typography",
        "modern_corporate_b2b",
        "luxury_editorial",
        "lifestyle_photography",
        "product_focused_commercial",
        "flat_illustration",
        "isometric_explainer",
        "high_impact_color_blocking",
        "retro_vintage",
        "futuristic_tech_dark",
        "glassmorphism_ui",
        "abstract_gradients",
        "infographic_data_driven",
        "quote_card_typography",
        "meme_style_social",
        "magazine_editorial",
        "cinematic_photography",
        "bold_geometric",
        "moody_atmospheric",
        "clean_tech",
        "hand_drawn_sketch",
        "neon_cyberpunk",
        "experimental_art",
        "brand_signature"
    ],

    "COMPOSITION_STYLE": [
        "center_focused",
        "rule_of_thirds",
        "symmetrical_clean",
        "asymmetrical_balance",
        "layered_depth",
        "framed_subject",
        "negative_space_heavy",
        "full_bleed_edge_to_edge",
        "collage_style"
    ]

}

# ---------------- THETA STORE ----------------
# theta per (dimension, value)
EMBEDDING_DIM = 3072  # 384 business + 384 topic

theta = defaultdict(lambda: np.zeros(EMBEDDING_DIM, dtype=np.float32))


# ---------------- UTILS ----------------

def softmax(scores):
    max_s = max(scores)
    exp = [math.exp(s - max_s) for s in scores]
    total = sum(exp)
    return [e / total for e in exp]


def build_context_vector(context):
    """
    Continuous context for generalization
    """
    return np.concatenate([
        context["business_embedding"],
        context["topic_embedding"]
    ])


# ---------------- ACTION SELECTION ----------------

def select_action(context):
    """
    context = {
      platform,
      time_bucket,
      business_embedding (384),
      topic_embedding (384)
    }
    """

    ctx_vec = build_context_vector(context)

    action = {}

    for dim, values in ACTION_SPACE.items():

        scores = []
        for v in values:
            # discrete preference
            H = db.get_preference(
                context["platform"],
                context["time_bucket"],
                dim,
                v
            )

            # continuous contribution
            score = H + np.dot(theta[(dim, v)], ctx_vec)
            scores.append(score)

        probs = softmax(scores)
        action[dim] = random.choices(values, probs)[0]

    return action, ctx_vec


# ---------------- LEARNING UPDATE ----------------

def update_rl(context, action, ctx_vec, reward, baseline,
              lr_discrete=0.05, lr_theta=0.01):
    print(f"🧠 Updating RL: reward={reward:.4f}, baseline={baseline:.4f}, advantage={reward - baseline:.4f}")
    ctx_vec = build_context_vector(context)
    advantage = reward - baseline

    for dim, val in action.items():
        print(f"   🎯 Updating action dimension: {dim}={val}")

        # 1️⃣ Discrete update (Supabase)
        db.update_preference(
            context["platform"],
            context["time_bucket"],
            dim,
            val,
            lr_discrete * advantage
        )

        # 2️⃣ Continuous update (theta)
        theta_update = lr_theta * advantage * ctx_vec
        theta[(dim, val)] += theta_update
        print(f"   📈 Theta update magnitude: {np.linalg.norm(theta_update):.6f}")


