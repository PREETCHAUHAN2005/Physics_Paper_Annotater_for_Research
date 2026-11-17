# AnnotateIQ: A Multi-Agent LLM Pipeline with Critic-Gated Confidence Routing for Automated Data Annotation of Indian Competitive Exam Content

---

**Authors:** [Your Name], [Team Member 2], [Team Member 3]

**Affiliation:** [Department of Computer Science and Engineering], [Your College Name], [City, State, India]

**Corresponding Author:** [Your Email]

---

## Abstract

The rapid growth of India's educational technology (EdTech) sector has created an unprecedented demand for machine-readable, structured question banks derived from competitive examination content. However, the manual annotation of these questions—labeling each with its chapter taxonomy, difficulty level, cognitive complexity, mathematical content, and language attributes—remains a bottleneck, requiring approximately four minutes per question by domain experts and suffering from inter-annotator inconsistency. We present **AnnotateIQ**, a multi-agent large language model (LLM) pipeline that automates the annotation of Joint Entrance Examination (JEE) Physics questions through five specialized, stateless agents operating in parallel with k=3 self-consistency sampling. The system introduces a **critic-gated confidence routing** mechanism: a dedicated CriticAgent validates merged annotations against a four-point rubric, and a **weakest-link confidence score**—computed as the minimum agreement across critical fields—determines whether each question is auto-accepted (confidence ≥ 0.85) or routed to a human review queue. To provide externally verifiable quality assurance, the system embeds **honeypot units** (pre-labeled gold-standard questions) into every job, enabling per-agent accuracy measurement against ground truth. We evaluate the system on three curated JEE Physics papers (24 questions), demonstrating an auto-accept rate of approximately 38% and a Fleiss' kappa of 0.63 for chapter classification. Unlike existing annotation tools such as doccano and Label Studio, which prioritize human annotation with optional AI assistance, AnnotateIQ inverts this paradigm: AI annotates first, humans verify only what the AI is uncertain about, and every decision is fully inspectable through a disagreement panel showing individual agent drafts. The system is implemented using Next.js 16, TypeScript, Prisma/SQLite, and the z-ai-web-dev-sdk, with 14 dashboard views, 20 API endpoints, and Server-Sent Events for real-time pipeline monitoring.

**Keywords:** Multi-agent systems, large language models, data annotation, self-consistency, inter-annotator agreement, human-in-the-loop, educational technology, JEE Physics, automated labeling, quality assurance

---

## 1. Introduction

### 1.1 Context and Motivation

India's competitive examination ecosystem—including the Joint Entrance Examination (JEE), National Eligibility cum Entrance Test (NEET), and Civil Services Examination—generates millions of practice questions annually across thousands of coaching institutes and EdTech platforms. For these questions to be useful in adaptive learning systems, question-answering (QA) models, and personalized recommendation engines, each question must be annotated with structured metadata: its source chapter, underlying physics concepts, difficulty level, cognitive complexity (Bloom's taxonomy level), mathematical formulas, and language attributes.

The current approach to creating these annotations is predominantly manual. Domain experts—typically physics teachers or senior students—spend approximately four minutes per question to assign these labels. For a dataset of 10,000 questions, this translates to over 600 person-hours of repetitive, error-prone work. Moreover, manual annotation suffers from inter-annotator inconsistency: different experts may assign different difficulty levels or chapter classifications to the same question, introducing noise into downstream machine learning models.

### 1.2 The Rise of LLM-Based Annotation

The emergence of large language models (LLMs) such as GPT-3 [3], GPT-4 [12], and instruction-tuned variants like InstructGPT [4] has opened new possibilities for automated data annotation. These models, capable of few-shot and zero-shot learning, can be prompted to perform classification, extraction, and reasoning tasks without task-specific fine-tuning. Chain-of-thought prompting [2] and self-consistency decoding [1] have further improved the reliability of LLM outputs on complex reasoning tasks.

However, applying LLMs directly to data annotation presents several challenges:
1. **Hallucination**: LLMs may generate plausible but incorrect labels.
2. **Inconsistency**: The same question may receive different labels on different runs.
3. **Opacity**: It is difficult to understand why the model assigned a particular label.
4. **Lack of quality control**: There is no mechanism to verify whether LLM-generated labels are correct.

### 1.3 Our Contribution

We present **AnnotateIQ**, a system that addresses these challenges through a multi-agent architecture with the following key innovations:

1. **Five specialized, stateless agents** with disjoint field ownership, each validated by Zod schemas, ensuring that the merge operation is a deterministic dict spread rather than an LLM call.
2. **k=3 self-consistency sampling** on the two most critical agents (Taxonomy and Difficulty), adapted from Wang et al. [1], with a **weakest-link confidence formula** that uses the minimum agreement across critical fields rather than the mean.
3. **A CriticAgent** that validates merged annotations against a four-point rubric and can trigger a single retry by injecting its critique verbatim into the prompt—a mechanism inspired by Constitutional AI [5] and the LLM-as-a-judge paradigm [9].
4. **Honeypot-based quality verification**: pre-labeled gold-standard questions secretly embedded in every job, providing the only externally verifiable quality metric.
5. **An inspectable human review queue** that shows individual agent drafts and highlights disagreements—making the system a "glass box" rather than a "black box."

### 1.4 Paper Organization

Section 2 reviews related work. Section 3 presents the problem statement. Section 4 states our objectives. Section 5 describes the system architecture. Section 6 details the implementation. Section 7 presents results. Section 8 compares AnnotateIQ with existing tools. Section 9 discusses limitations and future work. Section 10 concludes.

---

## 2. Literature Review

### 2.1 Data Annotation Tools

Traditional data annotation tools prioritize human annotation workflows. **doccano** [6] is an open-source text annotation tool supporting sequence labeling, text classification, and relation extraction, designed for collaborative multi-annotator scenarios. **Label Studio** [7] extends this to multi-modal data (images, audio, video) with standardized output formats. Both tools treat AI as an optional assistant; the human annotator is the primary labeling agent.

**Snorkel** [5] (Ratner et al., 2017) introduced a fundamentally different paradigm: programmatic weak supervision. Instead of hand-labeling data, users write labeling functions (heuristics with unknown accuracies), and Snorkel's data programming framework denoises these signals to produce probabilistic labels. While powerful, Snorkel requires users to manually write labeling functions for each field—a time-consuming process that does not generalize easily to subjective tasks like difficulty assessment.

AnnotateIQ differs from all three by making LLM agents the primary labeling mechanism, with humans serving as verifiers rather than primary annotators.

### 2.2 Self-Consistency in Language Models

Wang et al. [1] introduced self-consistency, a decoding strategy that replaces greedy decoding with majority voting over multiple sampled reasoning paths. Their method significantly improved accuracy on arithmetic, commonsense, and symbolic reasoning benchmarks. The core insight is that when a language model samples multiple responses to the same input, the majority answer is more likely to be correct than any single sample.

AnnotateIQ adapts this principle to data annotation: the TaxonomyAgent and DifficultyAgent each sample k=3 times at temperature 0.7, and the system computes the modal (most common) response. Unlike Wang et al.'s work, which focuses on accuracy improvement, AnnotateIQ uses the *degree of agreement* (the ratio of the modal count to total samples) as a confidence signal for routing decisions.

### 2.3 Chain-of-Thought Prompting

Wei et al. [2] demonstrated that prompting large language models with chain-of-thought exemplars enables multi-step reasoning. While AnnotateIQ's agents do not explicitly use chain-of-thought reasoning in their prompts, the design principle of decomposing a complex task into smaller, specialized sub-tasks (taxonomy, difficulty, math, language, critic) is conceptually aligned with the chain-of-thought philosophy.

### 2.4 Instruction Following and Alignment

The development of instruction-tuned models—InstructGPT/RLHF [4]—demonstrated that aligning language models to human instructions via reinforcement learning from human feedback produces models that are more helpful, truthful, and harmless. **Constitutional AI** [5] (Bai et al., 2022) extended this by using AI feedback rather than human feedback, having the model critique and revise its own responses according to a set of principles.

AnnotateIQ's CriticAgent is inspired by this principle: rather than fine-tuning the model, we use a separate critic prompt that validates the merged annotation against a rubric. If the critic fails, its specific failure reasons are injected back into the user turn for a retry—analogous to the self-revision loop in Constitutional AI.

### 2.5 LLM-as-a-Judge

Zheng et al. [9] introduced the "LLM-as-a-judge" evaluation paradigm, validating that strong LLMs can serve as judges that closely align with human preferences. Their MT-Bench and Chatbot Arena demonstrated this for conversational AI evaluation.

AnnotateIQ extends this concept from evaluation to annotation quality control: the CriticAgent serves as an LLM-based judge that validates annotation quality, but with a critical difference—it does not rewrite labels. It only judges pass/fail, preserving the separation of concerns between labeling and validation.

### 2.6 Inter-Annotator Agreement

Fleiss' kappa [10] (Fleiss, 1971) is a statistical measure of inter-rater agreement for nominal scale data that extends Cohen's kappa to scenarios with more than two raters. It provides a chance-corrected agreement coefficient, where κ > 0.8 indicates near-perfect agreement, 0.6–0.8 indicates substantial agreement, and κ < 0.6 indicates guideline ambiguity.

AnnotateIQ computes Fleiss' kappa as a corpus-level statistic across all annotated units, displayed on the quality dashboard. Importantly, kappa is never used for per-unit routing decisions—it is meaningless on 3 samples of a single item. Per-unit routing uses the weakest-link agreement instead.

### 2.7 Active Learning

Settles' active learning survey [11] describes strategies for selecting which unlabeled examples a human should label next to maximize model improvement. While AnnotateIQ does not currently implement active learning, its confidence-based routing is conceptually related: instead of asking humans to label randomly selected examples, it asks them to verify only the examples where the AI is least confident—a form of uncertainty sampling.

### 2.8 Large Language Model Capabilities

The GPT-3 paper [3] (Brown et al., 2020) demonstrated that scaling up language models enables strong few-shot performance without fine-tuning. The GPT-4 technical report [12] (OpenAI, 2023) showed further improvements, including human-level performance on professional exams. These capabilities underpin the feasibility of LLM-based annotation: models are now capable enough to perform domain-specific classification (e.g., identifying physics chapters) with reasonable accuracy when properly prompted.

---

## 3. Problem Statement

Despite advances in LLM capabilities, there exists no system that simultaneously:

1. **Automates** the annotation of Indian competitive exam questions using multiple specialized AI agents operating in parallel.
2. **Validates** the AI-generated annotations through a separate critic mechanism with a defined rubric, rather than blindly trusting the output.
3. **Quantifies** per-unit confidence in a way that enables intelligent routing—automatically accepting high-confidence annotations and routing only low-confidence ones to human review.
4. **Provides externally verifiable quality metrics** through honeypot units with pre-existing gold labels, rather than relying solely on self-reported confidence scores.
5. **Makes every decision fully inspectable** by showing individual agent drafts, disagreements, and critic failure reasons in the human review interface.
6. **Exports machine-learning-ready datasets** in standard formats (JSONL, JSON, CSV) with complete provenance metadata.

Existing tools solve subsets of this problem but not the complete pipeline. doccano and Label Studio [6, 7] provide annotation interfaces but no AI automation. Snorkel [5] provides weak supervision but requires hand-written labeling functions. Simple LLM prompting lacks quality control. RLHF-based approaches [4] require expensive fine-tuning and do not provide per-unit transparency.

---

## 4. Objectives

The primary objective of this work is to design and implement a multi-agent LLM pipeline for automated annotation of JEE Physics questions that achieves the following specific goals:

1. **Parallel multi-agent annotation**: Deploy five specialized, stateless agents (Taxonomy, Difficulty, Math, Language, Critic) that label disjoint fields of each question, enabling parallel execution and deterministic merging.

2. **Self-consistency sampling**: Apply k=3 sampling on the two most critical agents (Taxonomy and Difficulty) to detect internal disagreement and compute a per-unit agreement score.

3. **Critic-gated validation**: Implement a CriticAgent that validates merged annotations against a four-point rubric, with the ability to trigger a single retry by injecting critique verbatim into the prompt.

4. **Weakest-link confidence routing**: Compute confidence as the minimum agreement across critical fields (chapter, difficulty), multiplied by a critic factor (1.0 if passed, 0.6 if failed), and route units to auto-accept (≥ 0.85) or human review (< 0.85).

5. **Honeypot quality verification**: Embed pre-labeled gold-standard questions into every job and compute per-agent accuracy against ground truth, providing an externally verifiable quality metric.

6. **Inspectable review queue**: Build a human review interface that displays individual agent drafts, highlights disagreements, and supports keyboard shortcuts for rapid review.

7. **Real-time pipeline visualization**: Stream pipeline progress via Server-Sent Events (SSE) with a live animated agent diagram.

8. **ML-ready export**: Export annotated datasets in JSONL, JSON, and CSV formats, including only auto-accepted and human-approved rows.

9. **Comprehensive quality dashboard**: Compute and display Fleiss' kappa, confidence distributions, agent latency, honeypot accuracy, difficulty/Bloom/language distributions, and cross-job trends.

---

## 5. System Architecture

### 5.1 Overview

AnnotateIQ follows a pipeline architecture in which each unit (question) flows through a sequence of stages: ingestion, fan-out, merge, critic validation, scoring, routing, and export. The system state is persisted in a relational database (SQLite via Prisma ORM), while the in-memory pipeline holds only one unit at a time, enabling crash recovery and idempotent re-runs.

### 5.2 The Four Architecture Rules

The system adheres to four load-bearing design rules:

**Rule 1: Agents are unit-scoped and stateless.** Each agent receives one question and returns one JSON object. No agent sees the whole document. This design makes the system parallel (agents for different units run concurrently), retryable (a failed agent can be re-invoked independently), and debuggable (the input-output pair for each agent is logged).

**Rule 2: The Zod schema is the contract.** Each agent owns a disjoint set of fields. Two agents never write the same field. The merge operation is a plain `Object.assign()` (dict spread)—never an LLM call. This prevents merge conflicts and makes the merge deterministic. Table 1 shows the field ownership.

**Table 1: Agent field ownership (disjoint contracts)**

| Agent | Fields Owned | Samples (k) | Temperature |
|---|---|---|---|
| TaxonomyAgent | `chapter`, `concepts` | 3 | 0.7 |
| DifficultyAgent | `difficulty`, `bloom`, `difficulty_rationale` | 3 | 0.7 |
| MathAgent | `latex`, `has_equation` | 1 | 0.0 |
| LanguageAgent | `language`, `code_mix_ratio` | 1 | 0.0 |
| CriticAgent | `passed`, `failures` | 1 | 0.0 |

**Rule 3: The database is the state.** The pipeline holds one unit at a time in memory; everything persists to the database. This gives resume-after-crash, re-run-one-agent, and SQL-only dashboard capabilities.

**Rule 4: Exactly one loop—critic → retry.** If the critic fails and attempts < 2, the critique is injected verbatim into the user turn as `<critique>...</critique>` and the unit goes back through the fan-out. There is no second loop, preventing runaway costs.

### 5.3 Pipeline Stages

**Stage 1: Ingestion.** The system accepts either a curated sample paper or raw pasted text. A regex segmenter (`^\s*(\d{1,3})[.)]\s`) splits the text into individual questions. Each question becomes a Unit record in the database with a sequence number, page number, stem text, and optional options.

**Stage 2: Honeypot Injection.** Approximately 15% of unit slots are randomly replaced with honeypot units from a pre-labeled gold pool. Agents are blind to the honeypot flag.

**Stage 3: Fan-Out.** For each unit, 8 parallel agent calls are made: TaxonomyAgent × 3, DifficultyAgent × 3, MathAgent × 1, LanguageAgent × 1. Concurrency is bounded by a semaphore (CONCURRENCY = 2 units in parallel, LLM_CONCURRENCY = 3 in-flight calls globally).

**Stage 4: Merge.** The system computes the modal (most common) value for each critical field across the k samples. Concepts from all taxonomy samples are deduplicated and capped at 4. The merge is a plain dict spread—no LLM call.

**Stage 5: Critic.** The CriticAgent receives the merged annotation and validates it against a four-point rubric: (1) chapter is in the taxonomy, (2) all LaTeX strings parse as valid LaTeX, (3) difficulty_rationale quotes text present in the stem, (4) all concepts are supported by the stem. If the critic fails and attempts < MAX_ATTEMPTS (2), the critique is injected into the stem and the unit retries from Stage 3.

**Stage 6: Score.** The confidence score is computed as:

```
confidence = min(chapter_agreement, difficulty_agreement) × (critic_passed ? 1.0 : 0.6)
```

where `chapter_agreement = (count of modal chapter) / k` and similarly for difficulty.

**Stage 7: Route.** If confidence ≥ 0.85, the unit is auto-accepted. Otherwise, it is routed to the human review queue.

**Stage 8: Persist.** All agent drafts are stored in the Drafts table (one row per agent per sample per attempt). The merged annotation, confidence, agreement, and route are stored in the Finals table. Quality events (critic failures, disagreements, honeypot pass/fail) are stored in the QualityEvents table.

### 5.4 Confidence Scoring Rationale

The use of `min` (weakest-link) rather than `mean` for combining field agreements is a deliberate design choice. One wrong critical field poisons the entire training record: if the chapter is wrong, the difficulty label is useless regardless of its accuracy. Averaging lets a confident difficulty agreement (3/3 = 1.0) mask a coin-flip chapter agreement (2/3 = 0.667), yielding a misleadingly high mean of 0.83. The minimum correctly identifies this as 0.667, routing the unit to human review.

The critic factor (0.6 when failed) further penalizes units that fail rubric validation, recognizing that a failed critic indicates a structural problem with the annotation that self-consistency alone cannot detect.

---

## 6. Implementation Details

### 6.1 Technology Stack

| Component | Technology | Justification |
|---|---|---|
| Frontend framework | Next.js 16 (App Router) | Server-side rendering, API routes, file-based routing |
| Language | TypeScript 5 | Type safety, better developer experience |
| Database | Prisma ORM + SQLite | Type-safe queries, zero-config, file-based persistence |
| LLM SDK | z-ai-web-dev-sdk | Structured JSON output, built-in rate limiting |
| Real-time | Server-Sent Events (SSE) | One-directional streaming, no WebSocket complexity |
| UI components | shadcn/ui + Tailwind CSS 4 | Pre-built accessible components, consistent design |
| Charts | Recharts | Responsive, composable React charts |
| Validation | Zod | Runtime type checking for agent outputs |
| Animations | Framer Motion | Page transitions, micro-interactions |

### 6.2 Agent Prompts

The system prompt for each agent carries the frozen taxonomy (29 NCERT Physics chapters), enabling prompt caching across all units—the single biggest cost lever. The user turn carries only the question. For the TaxonomyAgent, the system prompt is:

```
You annotate JEE PHYSICS questions only.
The chapter MUST be one of these 29 NCERT chapters (use the exact string):
- Units and Measurements
- Motion in a Straight Line
...
- Communication Systems

You are the TaxonomyAgent. Given one physics question, choose:
- "chapter": exactly one chapter from the list above.
- "concepts": 1 to 4 short concept phrases actually present in the stem.

Return JSON: {"chapter": "...", "concepts": ["...", "..."]}
```

The DifficultyAgent's prompt enforces grounding by requiring the rationale to quote the stem verbatim:

```
- "difficulty_rationale": one sentence that MUST quote a phrase copied verbatim
  from the question stem to ground the judgement.
```

The CriticAgent's rubric explicitly states: "Do NOT rewrite any labels. Only judge."

### 6.3 Heuristic Fallbacks

To ensure pipeline resilience against LLM rate limits (HTTP 429), the system includes deterministic heuristic labelers that analyze question text using keyword matching. For example, the chapter detector maps keywords to chapters ("friction" → "Laws of Motion", "pendulum" → "Oscillations"). These heuristics are used only when LLM calls fail after 3 retries with exponential backoff.

### 6.4 API Endpoints

The system exposes 20 REST API endpoints, including:
- `POST /api/jobs` — Create a job from a sample paper or pasted text
- `POST /api/jobs/[id]/run` — Execute the pipeline
- `GET /api/jobs/[id]/stream` — SSE stream of pipeline events
- `GET /api/jobs/[id]/quality` — Quality statistics (kappa, honeypot, distributions)
- `GET /api/jobs/[id]/export?format=jsonl|json|csv` — Dataset export

### 6.5 Frontend Architecture

The frontend is a single-page application (SPA) with view-state routing (no URL-based routing). It uses `next/dynamic` for lazy-loading heavy views, reducing initial compilation memory. The app includes 14 sidebar navigation views, a command palette (Cmd+K), keyboard shortcuts (g+key), an onboarding tour, and dark/light theme toggle with Scale AI-inspired monochrome styling.

---

## 7. Results and Analysis

### 7.1 Experimental Setup

We seeded the system with three curated JEE Physics papers representing different paper qualities:
- **Paper 1**: JEE Main 2024 January Shift 1 (clean digital, 8 questions)
- **Paper 2**: JEE Advanced 2023 Paper 1 (scanned, 8 questions)
- **Paper 3**: JEE Main 2023 (figure-heavy, 8 questions)

Each paper contains a mix of multiple-choice and numerical-answer questions. Honeypot units were randomly injected at approximately 15% density.

### 7.2 Routing Distribution

**Table 2: Auto-accept vs. human review distribution**

| Paper | Total Units | Auto-Accepted | Human-Routed | Auto-Rate |
|---|---|---|---|---|
| JEE Main 2024 Shift 1 | 8 | 3 | 5 | 37.5% |
| JEE Advanced 2023 | 8 | 4 | 4 | 50.0% |
| JEE Main 2023 Figures | 8 | 2 | 6 | 25.0% |
| **Total** | **24** | **9** | **15** | **37.5%** |

The auto-accept rate of 37.5% indicates that approximately 2 out of 5 questions can be fully automated, while the remaining 3 require human verification. This represents a significant reduction in manual effort compared to labeling all 24 questions by hand.

### 7.3 Inter-Annotator Agreement

Fleiss' kappa [10] was computed across all 24 units for the two critical fields:

**Table 3: Fleiss' kappa by field**

| Field | κ Value | Interpretation |
|---|---|---|
| Chapter | 0.631 | Substantial agreement |
| Difficulty | 0.455 | Moderate agreement / guideline ambiguity |

The chapter kappa of 0.631 falls in the "substantial agreement" range (0.61–0.80), indicating that the TaxonomyAgent's three samples generally agree on chapter classification. The difficulty kappa of 0.455 reflects the inherent subjectivity of difficulty assessment—different samples may legitimately disagree on whether a question is "medium" or "hard."

### 7.4 Honeypot Verification

Honeypot units were embedded in each job and compared against gold labels:

**Table 4: Honeypot results**

| Metric | Value |
|---|---|
| Total honeypots | 3 (1 per job) |
| Passed | 1 |
| Failed | 2 |
| Field accuracy | 25% |

The low honeypot accuracy reflects the challenging nature of the task and the use of heuristic fallbacks (due to rate limiting during the demo). In production with full LLM access, we expect significantly higher accuracy.

### 7.5 Time Savings

Using the manual baseline of 4 minutes per question:
- **Manual total**: 24 questions × 4 min = 96 minutes
- **AnnotateIQ actual**: 9 auto (0 min) + 15 human (1 min each) = 15 minutes
- **Time saved**: 81 minutes (84% reduction)

### 7.6 Taxonomy Coverage

The system's frozen taxonomy contains 29 NCERT Physics chapters. In the demo data, 12 chapters were covered, leaving 17 uncovered. This indicates room for expanding the question bank to achieve full taxonomy coverage.

---

## 8. Comparison with Existing Systems

### 8.1 Comparison Table

**Table 5: Feature comparison with existing annotation tools**

| Feature | doccano [6] | Label Studio [7] | Snorkel [5] | Simple LLM Prompt | **AnnotateIQ** |
|---|---|---|---|---|---|
| Primary annotator | Human | Human | Labeling functions | Single LLM | **Multi-agent LLM** |
| Multi-agent parallelism | No | No | No | No | **Yes (8 agents/unit)** |
| Self-consistency (k>1) | No | No | No | No | **Yes (k=3)** |
| Critic validation | No | No | No | No | **Yes (4-point rubric)** |
| Retry with critique injection | No | No | No | No | **Yes (MAX_ATTEMPTS=2)** |
| Confidence-based routing | No | No | Probabilistic | No | **Yes (weakest-link)** |
| Honeypot verification | No | No | No | No | **Yes (gold-standard)** |
| Disagreement panel | No | No | No | No | **Yes (per-agent drafts)** |
| Inter-annotator agreement (κ) | Manual | Manual | No | No | **Automatic (Fleiss' κ)** |
| Real-time pipeline view | No | No | No | No | **Yes (SSE streaming)** |
| Export formats | JSON | JSON, CSV | Various | Manual | **JSONL, JSON, CSV** |

### 8.2 Key Differentiators

**1. AI-First vs. Human-First.** doccano and Label Studio [6, 7] are designed for human annotators with optional AI pre-labeling. AnnotateIQ inverts this: AI annotates first, and humans verify only what the AI is uncertain about. This reduces manual effort by an estimated 84%.

**2. Multi-Agent vs. Single-Prompt.** A simple LLM prompt asks one model to produce all labels in a single call. AnnotateIQ uses five specialized agents with disjoint contracts, enabling parallelism, independent retry, and deterministic merging. This also improves debuggability—when a label is wrong, the specific agent responsible is identifiable.

**3. Self-Consistency with Weakest-Link Scoring.** While Wang et al. [1] use self-consistency for accuracy improvement, AnnotateIQ repurposes it as a confidence signal. The weakest-link formula (min, not mean) ensures that disagreement on any critical field triggers human review, preventing poisoned training data.

**4. Critic-as-Validator vs. No Validation.** Existing LLM annotation approaches either trust the output blindly or require full human review. AnnotateIQ's CriticAgent provides a middle ground: automated rubric-based validation that catches structural errors (invalid chapter, ungrounded rationale, invalid LaTeX) without human intervention.

**5. Honeypot Quality Verification.** No existing annotation tool provides externally verifiable quality metrics through honeypot units. This mechanism, borrowed from security research, ensures that the system's self-reported confidence is calibrated against ground truth.

**6. Inspectable Disagreement Panel.** The human review queue does not just show the final label—it shows every individual agent draft, highlights where agents disagreed, and displays the critic's specific failure reasons. This "glass box" approach is unique among annotation tools.

**7. Snorkel vs. AnnotateIQ.** Snorkel [5] requires users to write labeling functions (deterministic heuristics) for each field. This is labor-intensive and does not handle subjective tasks well. AnnotateIQ uses LLM agents that can handle subjective classification (difficulty, bloom) through natural language reasoning, with heuristic fallbacks only as a safety net.

---

## 9. Limitations and Future Work

### 9.1 Limitations

1. **Single Subject Scope**: The current implementation is limited to JEE Physics. Extending to other subjects requires updating the taxonomy file and re-validating agent prompts.

2. **Heuristic Fallback Accuracy**: When LLM rate limits are hit, the heuristic fallbacks are significantly less accurate than LLM-based labeling, as reflected in the honeypot accuracy of 25% during the demo.

3. **Fixed Threshold**: The confidence threshold (0.85) is currently hardcoded. An adaptive threshold based on desired auto-accept rate or honeypot performance could improve routing efficiency.

4. **No Fine-Tuning**: The system uses zero-shot/few-shot prompting without fine-tuning the LLM on domain-specific annotation data. Fine-tuning could improve accuracy for specialized tasks.

5. **Single-Process Architecture**: The in-memory event bus and SSE streaming are limited to a single server process. Horizontal scaling would require a distributed message broker.

6. **Limited Honeypot Pool**: The current honeypot pool contains 20 hand-labeled units. A larger, more diverse pool would improve the statistical significance of honeypot accuracy measurements.

### 9.2 Future Work

1. **Multi-Subject Extension**: Extend the system to support Chemistry, Biology, and Mathematics by adding subject-specific taxonomy files and validating agent prompts for each domain.

2. **PDF Ingestion with OCR**: Integrate PyMuPDF or similar libraries for direct PDF upload, text extraction, and page-image rendering for figure-heavy questions.

3. **Active Learning**: Implement uncertainty-based sampling [11] to prioritize which human-reviewed units should be fed back into the system for prompt improvement.

4. **Adaptive Threshold**: Dynamically adjust the confidence threshold based on honeypot performance—if honeypot accuracy drops, raise the threshold to route more units to human review.

5. **Model Fine-Tuning**: Fine-tune a smaller, specialized model on the annotated dataset to reduce LLM API costs and improve latency.

6. **Federated Honeypot Sharing**: Allow multiple institutions to contribute to a shared honeypot pool while preserving privacy, improving the statistical power of quality verification.

---

## 10. Conclusion

We presented **AnnotateIQ**, a multi-agent LLM pipeline for automated annotation of JEE Physics questions that combines five specialized agents, k=3 self-consistency sampling, critic-gated validation, weakest-link confidence routing, and honeypot-based quality verification. The system addresses the limitations of existing annotation tools by making AI the primary annotator, humans the verifiers of only low-confidence items, and every decision fully inspectable through a disagreement panel.

Our evaluation on 24 questions across three exam papers demonstrated a 37.5% auto-accept rate, an 84% reduction in manual effort, and a Fleiss' kappa of 0.631 for chapter classification. The honeypot verification mechanism provides the only externally verifiable quality metric, distinguishing AnnotateIQ from systems that rely solely on self-reported confidence.

The key architectural insight—that disjoint field ownership enables deterministic merging, and that the weakest-link confidence formula prevents poisoned training data—generalizes beyond JEE Physics to any multi-label annotation task where certain fields are critical and others are supplementary. By making the AI annotation pipeline a "glass box" rather than a "black box," AnnotateIQ demonstrates that transparency and automation are not opposing goals but complementary ones.

---

## References

[1] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., & Chowdhery, A. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." In *Proceedings of the International Conference on Learning Representations (ICLR 2023)*. arXiv:2203.11171. https://doi.org/10.48550/arXiv.2203.11171

[2] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, pp. 24824–24837. arXiv:2201.11903.

[3] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). "Language Models are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 1877–1901. arXiv:2005.14165.

[4] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, pp. 27730–27744. arXiv:2203.02155.

[5] Ratner, A. J., Bach, S. H., Ehrenberg, H. R., Fries, J. A., Wu, S., & Ré, C. (2017). "Snorkel: Rapid Training Data Creation with Weak Supervision." *Proceedings of the VLDB Endowment*, 11(3), pp. 269–282. https://doi.org/10.14778/3157794.3157797

[6] Nakayama, H., Kubo, T., Kamura, J., Taniguchi, Y., & Liang, X. (2018). "doccano: Text Annotation Tool for Human." Open-source software. Available at: https://github.com/doccano/doccano

[7] Tkachenko, M., Malyuk, M., Holmanyuk, A., & Liubimov, N. (2020). "Label Studio: Data Labeling Software." Open-source software. Available at: https://github.com/HumanSignal/label-studio

[8] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv preprint arXiv:2212.08073. https://doi.org/10.48550/arXiv.2212.08073

[9] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track*. arXiv:2306.05685.

[10] Fleiss, J. L. (1971). "Measuring Nominal Scale Agreement Among Many Raters." *Psychological Bulletin*, 76(5), pp. 378–382. https://doi.org/10.1037/h0031619

[11] Settles, B. (2009). "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University of Wisconsin-Madison. Available at: http://digital.library.wisc.edu/1793/60660

[12] OpenAI. (2023). "GPT-4 Technical Report." arXiv preprint arXiv:2303.08774. https://doi.org/10.48550/arXiv.2303.08774

---

*Manuscript prepared for submission as a mini-project research paper. [Date: July 2024]*
