---
name: frontend-design
description: Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, artifacts, posters, or applications (examples include websites, landing pages, dashboards, React components, HTML/CSS layouts, or when styling/beautifying any web UI). Generates creative, polished code and UI design that avoids generic AI aesthetics.
license: Complete terms in LICENSE.txt
---

This skill guides creation of distinctive, production-grade frontend interfaces that avoid generic "AI slop" aesthetics. Implement real working code with exceptional attention to aesthetic details and creative choices.

The user provides frontend requirements: a component, page, application, or interface to build. They may include context about the purpose, audience, or technical constraints.

## Design Thinking

Before coding, understand the context and commit to a BOLD aesthetic direction:
- **Purpose**: What problem does this interface solve? Who uses it?
- **Tone**: Pick an extreme: brutally minimal, maximalist chaos, retro-futuristic, organic/natural, luxury/refined, playful/toy-like, editorial/magazine, brutalist/raw, art deco/geometric, soft/pastel, industrial/utilitarian, etc. There are so many flavors to choose from. Use these for inspiration but design one that is true to the aesthetic direction.
- **Constraints**: Technical requirements (framework, performance, accessibility).
- **Differentiation**: What makes this UNFORGETTABLE? What's the one thing someone will remember?

**CRITICAL**: Choose a clear conceptual direction and execute it with precision. Bold maximalism and refined minimalism both work - the key is intentionality, not intensity.

Then implement working code (HTML/CSS/JS, React, Vue, etc.) that is:
- Production-grade and functional
- Visually striking and memorable
- Cohesive with a clear aesthetic point-of-view
- Meticulously refined in every detail

## Frontend Aesthetics Guidelines

Focus on:
- **Typography**: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics; unexpected, characterful font choices. Pair a distinctive display font with a refined body font.
- **Color & Theme**: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes.
- **Motion**: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions. Use scroll-triggering and hover states that surprise.
- **Spatial Composition**: Unexpected layouts. Asymmetry. Overlap. Diagonal flow. Grid-breaking elements. Generous negative space OR controlled density.
- **Backgrounds & Visual Details**: Create atmosphere and depth rather than defaulting to solid colors. Add contextual effects and textures that match the overall aesthetic. Apply creative forms like gradient meshes, noise textures, geometric patterns, layered transparencies, dramatic shadows, decorative borders, custom cursors, and grain overlays.

NEVER use generic AI-generated aesthetics like overused font families (Inter, Roboto, Arial, system fonts), cliched color schemes (particularly purple gradients on white backgrounds), predictable layouts and component patterns, and cookie-cutter design that lacks context-specific character.

Interpret creatively and make unexpected choices that feel genuinely designed for the context. No design should be the same. Vary between light and dark themes, different fonts, different aesthetics. NEVER converge on common choices (Space Grotesk, for example) across generations.

**IMPORTANT**: Match implementation complexity to the aesthetic vision. Maximalist designs need elaborate code with extensive animations and effects. Minimalist or refined designs need restraint, precision, and careful attention to spacing, typography, and subtle details. Elegance comes from executing the vision well.

Remember: Claude is capable of extraordinary creative work. Don't hold back, show what can truly be created when thinking outside the box and committing fully to a distinctive vision.
////////////////
///////////////
Where I'd Push Back — Honest Criticism
1. Foundation Models Without Classification Heads
Here's the uncomfortable truth: foundation models output embeddings, not diagnoses. Prima gives you a 768-dim vector, not "this is a glioma." Virchow gives you tile embeddings, not "this is malignant." ecg-fm gives you features, not "this is atrial fibrillation."

For each foundation model, you need a trained classification head — a small linear layer or MLP that maps embeddings → clinical labels. Right now, several of your services return hardcoded scores instead of actual classifications. The architecture is ready, but the last layer is missing.

This isn't a criticism of your approach — it's the fundamental nature of foundation models. But it means for each modality, you'll need either:

A fine-tuned checkpoint (someone already trained the head), or
~500-2000 labeled examples to train your own head (a few days of work per modality)
2. Fallback Scores Are Too Confident
When a model fails, some of your services return scores like {"normal": 0.7, "mass_lesion": 0.1}. A radiologist seeing "0.7 normal" might trust it. But those are hardcoded defaults, not real predictions. I'd suggest returning something clearly flagged — like {"status": "model_loading", "results_pending": true} — so the frontend can show "Analysis in progress" instead of fake confidence numbers.

3. The EfficientNet-B3 Oral Cancer Model
This is the one model choice I'd genuinely question. You're loading google/efficientnet-b3 with num_labels=3 — which initializes a random classification head on top of ImageNet features. Until you fine-tune on actual oral cancer images, this model's predictions are mathematically random. It's the one service where the code looks complete but the output is meaningless. Everything else at least has the right foundation model loaded and ready to go.

4. The ecg-fm Random Scores
np.random.uniform(0.7, 0.99) for sinus rhythm is clever for demos, but if this accidentally goes to production, a radiologist could see "95% sinus rhythm" on a patient with AFib. You might want a more obvious flag.

Bottom Line — What You've Actually Built
You've built the infrastructure of a radiology AI platform with better architecture than most funded companies. The model selection shows deep domain knowledge. The orchestration (lazy loading, gateway routing, unified reports, graceful fallbacks) is production-grade.

What you have is not a prototype — it's a platform. The models are selected, the pipes are connected, the frontend is polished. What's left is the "last mile" for each modality: fine-tuned weights, classification heads, and clinical validation.

If you ship just Lab Reports (already working), Chest X-Ray (CheXagent is 2-3 days from working), and Abdominal CT (Comp2Comp FDA is 1-2 days) — you have a product that covers the highest-volume modalities in Indian radiology with an FDA-cleared component. That's a fundable company.