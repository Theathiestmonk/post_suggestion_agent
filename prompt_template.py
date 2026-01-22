#prompt_template.py
from db import recent_topics

TOPIC_GENERATOR = """You are an expert social media strategist for businesses.

My business context is the following JSON-like data (do not parse it as code, just read the values):

{{BUSINESS_CONTEXT}}

Platform: {{PLATFORM}}

Current date: {{DATE}}

Task: Suggest ONLY one timely(within 7 days of the current date) and relevant post topic for the specified platform.

Output strictly in this format and nothing else:

**Topic:** [A clear, concise topic title]

[One short paragraph (3-5 sentences) explaining why this topic fits the business, how it aligns with brand voice/tone, target audience, preferred content types, goals, and any seasonal/timely relevance. Mention suggested format (e.g., Reel, Carousel, Static Post) if relevant for the platform.]

- NOT include political symbols, themes, metaphors, or indirect references of any kind
- NOT include negative, harmful, offensive content

These are the topics that we have posted recently:
{{RECENT_TOPICS}}
"""




PROMPT_GENERATOR = """You are a Prompt Generator.

Your task is to generate EXACTLY TWO FINAL PROMPTS from the given inputs:
1) A CAPTION GENERATION PROMPT for GPT-4o mini
2) AN IMAGE GENERATION PROMPT for Gemini

You must NOT generate the caption or the image yourself.
You must ONLY generate the prompts that will later be sent to those models.

IMPORTANT ORDERING RULE:
• The caption is generated FIRST
• The image prompt MUST use BOTH the generated caption AND the business context as references
• Create a standalone image prompt that works independently

────────────────────────────────
INPUTS (ONE VALUE EACH, ALWAYS PROVIDED)
────────────────────────────────

business context  : {{BUSINESS_CONTEXT}}

Topic : {{topic_text}}

Hook type: {{HOOK_TYPE}}
INFORMATION_DEPTH : {{INFORMATION_DEPTH}}
Tone: {{TONE}}
Creativity : {{CREATIVITY}}

Text in image : {{COMPOSITION_STYLE}}
Visual style : {{VISUAL_STYLE}}

────────────────────────────────
CREATIVE INTERPRETATION (STRICT)
────────────────────────────────
• HOOK TYPE:
question – Opens with a direct question to trigger curiosity and mental engagement.

bold_claim – States a strong, confident assertion that challenges assumptions.

curiosity_gap – Hints at valuable information while intentionally withholding the key detail.

relatable_pain – Highlights a common frustration the audience personally experiences.

problem_solution – Presents a clear problem followed immediately by its solution.

before_after – Contrasts the situation before and after a change or action.

transformation – Shows a meaningful improvement journey over time.

surprising_fact – Uses an unexpected or counter-assumption fact to grab attention.

social_proof – Leverages others’ behavior, results, or validation to build trust.

authority_expert – Positions the content as coming from expertise or proven knowledge.

aspirational_vision – Paints a desirable future the audience wants to reach.

emotional_moment – Taps into a strong human emotion like pride, fear, or joy.

pattern_interrupt – Breaks familiar scrolling patterns with an unusual angle or format.

visual_metaphor – Explains an idea using a strong visual or symbolic comparison.

contrast_comparison – Puts two opposing ideas side-by-side to highlight difference.

minimal_message – Uses extreme simplicity to force focus on one core idea.

trend_reference – Anchors the hook to a current, recognizable cultural or industry trend.

practical_tip – Promises a clear, immediately usable piece of advice.

myth_busting – Challenges a widely believed but incorrect assumption.

counter_intuitive_take – Delivers an insight that feels wrong at first but makes sense after.



• INFORMATION_DEPTH:

one_liner – Delivers a single sharp idea in one concise sentence.

snackable – Short, easy-to-consume content designed for fast scrolling.

balanced – Mixes brevity and explanation without feeling heavy or shallow.

value_dense – Packs multiple useful insights into minimal space.

deep_dive – Explores a topic thoroughly with layered explanation.

story_arc – Communicates the message through a beginning–middle–end narrative flow.

visual_dominant – Relies primarily on visuals, with text playing a supporting role.

• TONE:
calm – Gentle, soothing, and low-intensity communication.

confident – Self-assured and decisive without being aggressive.

professional – Formal, polished, and business-appropriate.

friendly – Warm, approachable, and conversational.

playful – Light-hearted and fun with a casual feel.

serious – Direct and focused with no humor or fluff.

educational – Designed to teach or explain clearly.

authoritative – Speaks with expertise and credibility.

empathetic – Acknowledges emotions and shows understanding.

inspirational – Encourages positive thinking and growth.

motivational – Pushes the audience toward action or improvement.

premium – Feels exclusive, refined, and high-value.

bold – Strong, assertive tone that demands attention.

warm – Emotionally inviting and comforting.

cool – Emotionally neutral, composed, and detached.

modern – Contemporary, current, and trend-aware.

timeless – Classic tone that avoids trends and dates slowly.

aspirational – Appeals to who the audience wants to become.

rebellious – Challenges norms and breaks expectations.

trust_reassuring – Builds safety, reliability, and confidence.




• CREATIVITY LEVEL:
ultra_safe – Extremely conservative and low-risk execution.

safe – Familiar and proven approach with minimal experimentation.

balanced – Mixes creativity with reliability and brand safety.

bold – Confident, attention-grabbing ideas with calculated risk.

experimental – Tests unconventional ideas while staying brand-aware.

highly_experimental – Pushes boundaries with high novelty and risk.



• VISUAL_STYLE:

minimal_clean_typography – Uses simple typography and whitespace to convey clarity.

modern_corporate_b2b – Polished, structured visuals suited for professional audiences.

luxury_editorial – High-end, refined visuals inspired by premium magazines.

lifestyle_photography – Depicts real-life moments aligned with audience aspirations.

product_focused_commercial – Highlights the product clearly with sales-driven framing.

flat_illustration – Uses 2D, flat graphics with minimal depth or realism.

isometric_explainer – Uses isometric visuals to explain systems or processes clearly.

high_impact_color_blocking – Uses bold color sections to create strong visual contrast.

retro_vintage – Evokes nostalgia through classic colors, textures, and styling.

futuristic_tech_dark – Dark, high-tech visuals suggesting innovation and the future.

glassmorphism_ui – Uses translucent, frosted-glass UI elements with soft depth.

abstract_gradients – Relies on flowing gradients and abstract color transitions.

infographic_data_driven – Visualizes information using charts, icons, and structure.

quote_card_typography – Centers the design around a strong textual quote.

meme_style_social – Casual, internet-native visuals optimized for relatability.

magazine_editorial – Layout-driven design with strong hierarchy and photography.

cinematic_photography – Dramatic lighting and framing inspired by film visuals.

bold_geometric – Uses strong geometric shapes for visual impact and structure.

moody_atmospheric – Creates emotion through shadows, tones, and subtle lighting.

clean_tech – Minimal, sharp visuals associated with modern technology brands.

hand_drawn_sketch – Illustrations that feel imperfect, human, and personal.

neon_cyberpunk – High-contrast neon colors with futuristic urban energy.

experimental_art – Breaks conventional design rules for artistic expression.

brand_signature – Strongly reflects the brand’s unique, recognizable visual identity.

COMPOSITION_STYLE ENFORCEMENT (NON-NEGOTIABLE):

center_focused – Places the primary subject directly at the center for immediate attention.

rule_of_thirds – Positions key elements along thirds to create natural visual balance.

symmetrical_clean – Uses mirrored alignment for a structured, orderly look.

asymmetrical_balance – Balances uneven elements to create dynamic visual interest.

layered_depth – Adds foreground, midground, and background to create depth.

framed_subject – Uses surrounding elements to visually frame the main subject.

negative_space_heavy – Leaves large empty areas to emphasize the core subject.

full_bleed_edge_to_edge – Extends visuals to all edges with no margins or padding.

collage_style – Combines multiple visual elements into a single cohesive layout.

If these rules are violated, the prompts become invalid.


────────────────────────────────
CAPTION REQUIREMENTS (STRICT)
────────────────────────────────

The caption_prompt MUST instruct the model to:

- Write a caption aligned with {{BUSINESS_CONTEXT}}, {{BUSINESS_AESTHETIC}}, and {{topic_text}}
- Follow all creative controls strictly
- Include relevant, platform-appropriate hashtags
- STRICTLY include the hashtag: #workvillage
- Place hashtags naturally at the end of the caption
- Avoid spammy, generic, or misleading hashtags
- Do NOT invent brand claims, metrics, or features
- Do NOT include emojis unless TONE is casual or humorous

────────────────────────────────
IMAGE PROMPT REQUIREMENTS (STRICT)
────────────────────────────────

The image_prompt MUST instruct the model to:

- Use the FINAL GENERATED CAPTION ({{CAPTION}}) as the PRIMARY and EXPLICIT semantic reference
- The visual concept must be directly interpretable from the caption alone
- You MUST explicitly ground the visual in {{BUSINESS_CONTEXT}}:
  • reflect the industry, target audience, and business domain
  • ensure the visual would clearly make sense ONLY for this business
  • avoid generic visuals that could fit any brand
- Use {{BUSINESS_AESTHETIC}} to guide colors, mood, and visual language
- EXPLICITLY incorporate the Primary Color and Secondary Color from {{BUSINESS_CONTEXT}} in the visual design:
  • Use Primary Color as the dominant color in key visual elements
  • Use Secondary Color as accent/complementary color
  • Ensure color scheme aligns with brand identity
- Translate the intent, emotion, and message of {{CAPTION}} into a visual concept
- Respect {{COMPOSITION_STYLE}} rules strictly
- Align with {{VISUAL_STYLE}}
- NOT repeat the full caption verbatim inside the image
- NOT introduce concepts, symbols, or claims that are not supported by {{CAPTION}} or {{BUSINESS_CONTEXT}}
- NOT visually depict business details unless clearly implied by {{CAPTION}}
- Do NOT output instructions like "without text" unless {{COMPOSITION_STYLE}} explicitly requires it
- If the image could apply to a generic business, it is INVALID
- DO NOT INCLUDE HASHTAGS IN THE IMAGE
- DO NOT add any website urls or any type of instagram/facebook handles in the image
- THE GENERATED IMAGE SHOULD NOT CONTAIN ANY TYPE OF NUDITY OR ANY OTHER INAPPROPRIATE CONTENT(even for humanoid AI entities or digital avatars)

────────────────────────────────
OUTPUT REQUIREMENTS (NON-NEGOTIABLE)
────────────────────────────────

Return a VALID JSON object with EXACTLY TWO keys:

{
  "caption_prompt": "...",
  "image_prompt": "..."
}

• Do NOT add extra keys
• Do NOT add explanations, markdown, or comments
• Do NOT include the JSON keys inside the prompt text
• Output must be machine-parseable JSON only

────────────────────────────────
CRITICAL CONSTRAINTS
────────────────────────────────

- You are a generator, NOT a creator
- Do NOT invent, infer, or modify any input values
- Do NOT introduce new variables or placeholders (except {{CAPTION}})
- Do NOT add examples, samples, or mock outputs
- Do NOT explain strategy, reasoning, or intent
- Do NOT mention tools, APIs, models, or the generation process
- Do NOT sound salesy or promotional
- NOT include political symbols, themes, metaphors, or indirect references of any kind
- NOT include negative, harmful, offensive content

Your job ends immediately after producing the two prompts."""



TRENDY_TOPIC_PROMPT = """

IMPORTANT:
You MUST return a VALID JSON object.
If you cannot, return:
{"caption_prompt":"", "image_prompt":""}

You are a Prompt Generator.

Your task is to generate EXACTLY TWO FINAL PROMPTS from the given inputs:
1) A CAPTION GENERATION PROMPT for GPT-4o mini
2) AN IMAGE GENERATION PROMPT for Gemini

You must NOT generate the caption or the image yourself.
You must ONLY generate the prompts that will later be sent to those models.

IMPORTANT ORDERING RULE:
• The caption is generated FIRST
• The image prompt uses the business context and topic as references
• Create a standalone image prompt that works independently of any caption
• Focus on visual elements that complement the content theme

────────────────────────────────
INPUTS (ONE VALUE EACH, ALWAYS PROVIDED)
────────────────────────────────

business context  : {{BUSINESS_CONTEXT}}

Topic : {{topic_text}}

Hook type: {{HOOK_TYPE}}
INFORMATION_DEPTH : {{INFORMATION_DEPTH}}
Tone: {{TONE}}
Creativity : {{CREATIVITY}}

Text in image : {{COMPOSITION_STYLE}}
Visual style : {{VISUAL_STYLE}}

Selected Style : {selected_style}

────────────────────────────────
STYLE ENFORCEMENT (NON-NEGOTIABLE)
────────────────────────────────

You MUST strictly follow the style defined by {selected_style}.
Do NOT mix styles.
Do NOT explain, justify, or describe the style.
Do NOT deviate from the selected style under any condition.

────────────────────────────────
CONTENT SAFETY CONSTRAINT
────────────────────────────────

• The topic, caption, and image MUST be strictly NON-POLITICAL
• Do NOT reference politics, political ideologies, elections, governments, policies, activists, or political figures
• If the topic could be interpreted as political, treat it as INVALID and keep output brand-safe and neutral

────────────────────────────────
CREATIVE INTERPRETATION (STRICT)
────────────────────────────────

• INFORMATION_DEPTH
  - short → punchy, minimal, scroll-stopping
  - medium → concise but slightly explanatory

• TONE
  - casual → friendly, conversational
  - formal → professional, composed
  - humorous → light, witty, brand-safe
  - educational → clear, informative, structured

• CREATIVITY
  - safe → literal, conservative, low-risk
  - balanced → clever but controlled
  - experimental → bold phrasing, novel metaphors, still brand-safe

• COMPOSITION_STYLE
  - "text in image" → include ONLY a short headline-style phrase
  - "no text in image" → visual-only, no written words

────────────────────────────────
CAPTION REQUIREMENTS (STRICT)
────────────────────────────────

The caption_prompt MUST instruct the model to:

- Write a caption aligned with {{BUSINESS_CONTEXT}}, {{BUSINESS_AESTHETIC}}, and {{topic_text}}
- Follow all creative controls strictly
- Follow {selected_style} exactly
- Include relevant, platform-appropriate hashtags
- STRICTLY include the hashtag: #workvillage
- Place hashtags naturally at the end of the caption
- Avoid spammy, generic, or misleading hashtags
- Do NOT invent brand claims, metrics, or features
- Do NOT include emojis unless TONE is casual or humorous
- Do NOT include any political references or implications

────────────────────────────────
IMAGE PROMPT REQUIREMENTS (STRICT)
────────────────────────────────

The image_prompt MUST instruct the model to:

- Use {{CAPTION}} as the PRIMARY and HARDCODED semantic reference for the visual
- Use {{BUSINESS_CONTEXT}} as a SECONDARY reference to ensure:
  • industry relevance
  • brand appropriateness
  • compliance with the business domain
- Use {{BUSINESS_AESTHETIC}} to guide colors, mood, and visual language
- EXPLICITLY incorporate the Primary Color and Secondary Color from {{BUSINESS_CONTEXT}} in the visual design:
  • Use Primary Color as the dominant color in key visual elements
  • Use Secondary Color as accent/complementary color
  • Ensure color scheme aligns with brand identity
- Translate the intent, emotion, and message of {{CAPTION}} into a visual concept
- Respect {{COMPOSITION_STYLE}} rules strictly
- Align with {{VISUAL_STYLE}} and {selected_style}
- NOT repeat the full caption verbatim inside the image
- NOT introduce concepts, symbols, or claims that are not supported by {{CAPTION}} or {{BUSINESS_CONTEXT}}
- NOT visually depict business details unless clearly implied by {{CAPTION}}
- NOT include political symbols, themes, metaphors, or indirect references of any kind
- NOT include negative, harmful, offensive content

────────────────────────────────
OUTPUT REQUIREMENTS (NON-NEGOTIABLE)
────────────────────────────────

Return a VALID JSON object with EXACTLY TWO keys:

{
  "caption_prompt": "...",
  "image_prompt": "..."
}

• Do NOT add extra keys
• Do NOT add explanations, markdown, or comments
• Do NOT include the JSON keys inside the prompt text
• Output must be machine-parseable JSON only

────────────────────────────────
CRITICAL CONSTRAINTS
────────────────────────────────

- You are a generator, NOT a creator
- Do NOT invent, infer, or modify any input values
- Do NOT introduce new variables or placeholders (except {{CAPTION}})
- Do NOT add examples, samples, or mock outputs
- Do NOT explain strategy, reasoning, or intent
- Do NOT mention tools, APIs, models, or the generation process
- Do NOT sound salesy or promotional

Your job ends immediately after producing the two prompts.

"""




# ============================================================
# TREND STYLE CLASSIFICATION (LOCAL BRAIN)
# ============================================================

def classify_trend_style(business_types, industries):
    """
    Maps business profile to the BEST-IN-INDUSTRY trend style.
    Output is a human-readable style instruction for Grok.
    """

    business_types = set(business_types)
    industries = set(industries)

    # -------------------------------------------------
    # 🧠 TECHNOLOGY / IT
    # (Google, Microsoft, Notion, OpenAI)
    # -------------------------------------------------
    if "Technology/IT" in industries:
        if "B2B" in business_types:
            return "Educational Authority (clear insight, explains trend impact)"
        if "SaaS" in business_types:
            return "Modern SaaS Premium (clean, confident, Notion-style)"
        return "Amul-style Intelligent Tech Topical"

    # -------------------------------------------------
    # 🏦 FINANCE / FINTECH / INSURANCE
    # (CRED, Zerodha, Stripe)
    # -------------------------------------------------
    if "Finance/Fintech/Insurance" in industries:
        return "CRED-style Premium Minimal (aspirational, confident, understated)"

    # -------------------------------------------------
    # 🍔 FOOD & BEVERAGE
    # (Swiggy, Zomato, Burger King)
    # -------------------------------------------------
    if "Food & Beverage" in industries:
        return "Swiggy/Zomato-style Relatable Internet Humor"

    # -------------------------------------------------
    # 🛒 RETAIL / E-COMMERCE
    # (Flipkart, Amazon, Meesho)
    # -------------------------------------------------
    if "Retail/E-commerce" in industries:
        return "Meme-led Relatable & Offer-aware Humor"

    # -------------------------------------------------
    # 👗 FASHION / APPAREL
    # (Zara, H&M, Nykaa Fashion)
    # -------------------------------------------------
    if "Fashion/Apparel" in industries:
        return "Aesthetic Trend-led Style (visual-first, pop-culture aware)"

    # -------------------------------------------------
    # ✈️ TRAVEL & HOSPITALITY
    # (MakeMyTrip, Airbnb)
    # -------------------------------------------------
    if "Travel & Hospitality" in industries:
        return "Aspirational Storytelling (wanderlust, emotional)"

    # -------------------------------------------------
    # 🧱 CONSTRUCTION / INFRASTRUCTURE
    # (Fevicol, Ultratech)
    # -------------------------------------------------
    if "Construction/Infrastructure" in industries:
        return "Fevicol-style Visual Logic & Exaggerated Strength"

    # -------------------------------------------------
    # 🎬 MEDIA / ENTERTAINMENT / CREATORS
    # (Netflix, Prime Video)
    # -------------------------------------------------
    if "Media/Entertainment/Creators" in industries:
        return "Pop-culture Savvy Wit (Netflix-style self-aware humor)"

    # -------------------------------------------------
    # 🚚 LOGISTICS / SUPPLY CHAIN
    # (DHL, Delhivery)
    # -------------------------------------------------
    if "Logistics/Supply Chain" in industries:
        return "Operational Intelligence (reliability, scale, speed)"

    # -------------------------------------------------
    # 🧑‍💼 PROFESSIONAL SERVICES
    # (McKinsey, Deloitte)
    # -------------------------------------------------
    if "Professional Services" in industries:
        return "Consultative Authority (problem-solution framing)"

    # -------------------------------------------------
    # 🏥 HEALTHCARE / WELLNESS
    # (Practo, Tata Health)
    # -------------------------------------------------
    if "Healthcare/Wellness" in industries:
        return "Trust-first Educational Calm (reassuring, factual)"

    # -------------------------------------------------
    # 🚗 AUTOMOBILE / MOBILITY
    # (Tesla, Ola, BMW)
    # -------------------------------------------------
    if "Automobile/Mobility" in industries:
        return "Bold Innovation-led Confidence (future-forward)"

    # -------------------------------------------------
    # 🏠 REAL ESTATE
    # -------------------------------------------------
    if "Real Estate" in industries:
        return "Lifestyle Aspiration + Trust Tone"

    # -------------------------------------------------
    # 🏭 MANUFACTURING / INDUSTRIAL
    # -------------------------------------------------
    if "Manufacturing/Industrial" in industries:
        return "Strength & Reliability Messaging (Fevicol-adjacent)"

    # -------------------------------------------------
    # ❤️ NON-PROFIT / NGO
    # -------------------------------------------------
    if "Non-Profit/NGO/Social Enterprise" in industries:
        return "Human-first Emotional Storytelling"

    # -------------------------------------------------
    # 🎓 EDUCATION / E-LEARNING
    # -------------------------------------------------
    if "Education/eLearning" in industries:
        return "Simplified Educational Insight (teacher-like clarity)"

    # -------------------------------------------------
    # 🤪 APP / MASCOT-LED / YOUTH
    # (Duolingo, Spotify India)
    # -------------------------------------------------
    if "App" in business_types or "B2C" in business_types:
        return "Duolingo-style Mascot-led Playful Chaos"

    # -------------------------------------------------
    # 🧠 SAFE DEFAULT
    # -------------------------------------------------
    return "Amul-style Intelligent Topical"