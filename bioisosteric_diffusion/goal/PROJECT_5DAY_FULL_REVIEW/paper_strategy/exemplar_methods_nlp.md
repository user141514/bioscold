# Exemplar Methods Sections from Classic NLP Papers: A Comparative Analysis

## Purpose

This document dissects the Methods sections of four landmark NLP papers to extract **writing craft** — not content summaries, but structural and stylistic techniques that make a Methods section work. The goal is to derive actionable patterns for writing the Methods section of a computational chemistry paper.

---

## 1. Attention Is All You Need (Vaswani et al., 2017)

**Section structure:** Section 3 (Model Architecture) + Section 5 (Training)

### 1.1 Section Organization

```
3. Model Architecture                        [2 paragraphs intro]
    3.1 Encoder and Decoder Stacks           [2 paragraphs]
    3.2 Attention
        3.2.1 Scaled Dot-Product Attention   [1 paragraph + 1 equation]
        3.2.2 Multi-Head Attention           [2 paragraphs + 2 equations]
        3.2.3 Applications of Attention      [1 paragraph]
    3.3 Position-wise Feed-Forward Networks  [1 paragraph + 1 equation]
    3.4 Embeddings and Softmax               [1 paragraph]
    3.5 Positional Encoding                  [2 paragraphs + 2 equations]
    
5. Training
    5.1 Training Data and Batching           [1 paragraph]
    5.2 Hardware and Schedule                [1 paragraph]
    5.3 Optimizer                            [1 paragraph + 1 equation]
    5.4 Regularization                       [1 paragraph]
```

### 1.2 How They Describe Architecture

The paper uses **one massive architectural diagram (Figure 1) as the primary description vehicle**, then walks through each component with minimal prose. The strategy is:

1. **Figure first:** Figure 1 shows the entire encoder-decoder stack with attention, residual connections, and masking at a glance.
2. **Prose as annotation:** The text does not re-describe the figure; it annotates it. Typical sentence:
   > "The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network."
3. **Equations for the core innovation only:** Three key equations total in Section 3 (attention, multi-head composition, FFN). Everything else is prose or the diagram.
4. **No derivations, no background explanations.** The paper assumes the reader knows layer normalization, residual connections, and softmax attention. This is a deliberate choice: the novelty is the architecture, not the individual components.

### 1.3 How They Describe Training

Training details are in a separate section (Section 5), not mixed with architecture. Each subsection is exactly one paragraph:

- **Optimizer:** Exact Adam hyperparameters (beta1=0.9, beta2=0.98, epsilon=1e-9) with the LR schedule formula:
  > `lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))`
- **Regularization:** "We apply residual dropout [...] to the output of each sub-layer, before it is added to the sub-layer input and normalized. [...] We also apply label smoothing of value epsilon_ls = 0.1."
- **Hardware:** "We trained our models on one machine with 8 NVIDIA P100 GPUs."

### 1.4 Ablations and Variants

The Transformer paper presents model variants **inline in the section structure** rather than in a separate ablation section. The "Base" and "Big" models are defined during the architecture description (N=6 for both, but d_model and d_ff differ). The comparison of attention vs. RNN vs. CNN is Section 4 ("Why Self-Attention"), which is essentially a standalone inductive-bias analysis with Table 1.

### 1.5 Prose-to-Equations-to-Tables/Figures Ratio

| Element | Approx. proportion of Section 3+5 |
|---------|----------------------------------|
| Prose | ~70% |
| Equations | ~10% (4 equations total in Sections 3+5) |
| Figures | ~15% (one large diagram, one table) |
| Tables | ~5% (Table 1: complexity comparison, Table 2: BLEU results) |

### 1.6 Length

Section 3 is about **4 pages**. Section 5 is about **2 pages**. Total Methods: ~6 pages in the published paper.

### 1.7 What Makes This Reproducible

1. **Every hyperparameter is stated explicitly** in numerical form — not "we used Adam" but "Adam with beta1=0.9, beta2=0.98, epsilon=1e-9."
2. **The LR schedule is a mathematical formula** — any implementer can code it precisely.
3. **Hardware and training time are given** ("100,000 steps, 12 hours on 8 P100s").
4. **Data preprocessing is described** (BPE with 37K tokens, batched by approximate sequence length).
5. **Legal reproducibility:** The authors released the TensorFlow code.

### 1.8 Signature Writing Techniques

- **Short, declarative topic sentences** at the start of each subsection. Example: "An attention function can be described as mapping a query and a set of key-value pairs to an output."
- **Every architectural choice comes with a reason**, even if just a sentence. Example: "We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."
- **Comparative framing:** Each component is justified against the alternative. "The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1/sqrt(d_k)."
- **The architecture diagram does the heavy lifting.** The paper invests heavily in one excellent figure rather than many mediocre ones.

---

## 2. BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)

**Section structure:** Section 3 (BERT)

### 2.1 Section Organization

```
3. BERT                                              [2 paragraphs intro]
    [Model Architecture paragraph]                   
    [Input/Output Representations paragraph]         
    3.1 Pre-training BERT                            
        Task #1: Masked LM                           [3 paragraphs]
        Task #2: Next Sentence Prediction (NSP)      [2 paragraphs]
        [Pre-training data paragraph]                
    3.2 Fine-tuning BERT                             [2 paragraphs]
    3.3 Comparison of Pre-training Procedures        [1 paragraph + Table]
```

### 2.2 How They Describe Architecture

BERT's approach is the opposite of the Transformer: **architecture is described in prose as a delta from the Transformer**, with no new architecture diagram. The key description:

> "BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017)."

The novelty is in the **pre-training tasks**, not the architecture, so that is where the prose effort goes. The architecture section is just one paragraph plus a table comparing BERT_BASE to BERT_LARGE. This is a deliberate strategic choice: don't re-describe what people already know.

The architecture table is clean and minimal:

| Model | L | H | A | Params |
|-------|---|---|---|--------|
| BERT_BASE | 12 | 768 | 12 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 340M |

With the clarifying note: "BERT_BASE was chosen to have the same model size as OpenAI GPT for comparison purposes."

### 2.3 How They Describe Training (Pre-training)

The pre-training procedure is the paper's core contribution, and the writing reflects this. The structure is **problem-first**:

**For Masked LM:**
1. State the problem: "Standard conditional language models can only be trained left-to-right or right-to-left. [...] such a unidirectional approach is sub-optimal."
2. Propose the solution: "To make this more straightforward, we simply mask some percentage of the input tokens at random, and then predict those masked tokens."
3. Address the mismatch problem: "The [MASK] token does not appear in fine-tuning," so they introduce the 80/10/10 strategy.
4. State the loss: "The final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary [...] compute a cross-entropy loss."

**For NSP:**
1. State what's missing: "Many important downstream tasks [...] are based on understanding the relationship between two sentences, which is not directly captured by language modeling."
2. Propose: "We pre-train a binarized next sentence prediction task."
3. Give protocol: "50% of the time B is the actual next sentence that follows A [...] 50% of the time it is a random sentence."

Then all hyperparameters are listed in one dense paragraph at the end of Section 3.1:
> "The model was trained with a batch size of 256 sequences [...] for 1,000,000 steps [...] Adam with learning rate of 1e-4, beta1=0.9, beta2=0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate."

### 2.4 Ablations

BERT's ablations are in a separate Section 4 (not in the Methods section), with clear subsection structure:

- **4.1 Effect of Pre-training Tasks:** Compares No NSP, LTR & No NSP, + BiLSTM. Each with a 1-2 sentence result summary. Key sentences:
  > "Removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1."
  > "The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD."
- **4.2 Effect of Model Size:** Six model sizes, from L=3 to L=24. Clear table. Key finding:
  > "Larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples."
- **4.3 Feature-based Approach with BERT:** Tests feature extraction vs. fine-tuning.

### 2.5 Prose-to-Equations-to-Tables/Figures Ratio (Section 3 only)

| Element | Proportion |
|---------|-----------|
| Prose | ~80% |
| Equations | ~3% (only one formal equation — softmax for SQuAD span prediction) |
| Figures | ~10% (Figure 1: architecture overview, Figure 2: input representation) |
| Tables | ~7% (Model size comparison table) |

BERT is the **most prose-heavy** of the four papers in its Methods section. It uses very few equations, preferring clear English explanations and bullet-style lists (the 80/10/10 rule is stated in prose, not as math).

### 2.6 Length

Section 3 is about **5 pages** (including one figure of the input representation and one table).

### 2.7 What Makes This Reproducible

1. **Every hyperparameter is in one dense paragraph** — optimizer, LR, warmup steps, decay, batch size, total steps, dropout, activation function.
2. **Hardware is stated:** "BERT_BASE was trained on 4 Cloud TPUs in Pod configuration (16 TPU chips total)."
3. **Training time:** "Each pre-training run took 4 days."
4. **Data sources are quantified:** "BooksCorpus (800M words) and English Wikipedia (2,500M words)."
5. **Fine-tuning hyperparameters are given as a search range:** "batch size of 16 or 32, learning rate of 5e-5, 4e-5, 3e-5, or 2e-5, and 2-4 epochs."
6. **Code release:** The authors released the TensorFlow code.

### 2.8 Signature Writing Techniques

- **"To overcome this, we..." structure:** Every design decision has a clear motivation stated as "Problem. To address this, we propose X."
- **Explicit comparisons to prior art:** Constantly positioned against ELMo and OpenAI GPT. Example: "the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left."
- **Self-aware limitations:** The paper admits the MLM/fine-tuning mismatch and explains exactly why the 80/10/10 heuristic mitigates it.
- **Forward references for depth:** "We compare various masking strategies in Appendix C.2."
- **Concrete numbers embedded in prose** rather than relying solely on tables for key facts.

---

## 3. GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)

**Section structure:** Section 2 (Approach)

### 3.1 Section Organization

```
2. Approach                                          [1 paragraph intro]
    2.1 Training Dataset                              [1 paragraph]
    2.2 Input Representation                         [1 paragraph]
    2.3 Model Architecture                           [1 paragraph + table]
        4 model sizes: Small (117M), Medium (345M), 
                       Large (762M), XL (1.5B)
```

The entire "Methods" section is **remarkably short** — about 2.5 pages. This is the most concise of the four papers.

### 3.2 How They Describe Architecture

GPT-2 also uses the **delta-from-prior-work** strategy, but even more aggressively than BERT. The architecture description is one paragraph:

> "Our model largely follows the details of the OpenAI GPT model with a few modifications."

Then the modifications are listed as bullet changes:
1. Layer normalization moved to the input of each sub-block (pre-activation)
2. Additional layer normalization after the final self-attention block
3. Modified initialization: residual layer weights scaled by 1/sqrt(N)
4. Vocabulary expanded to 50,257 tokens
5. Context size expanded from 512 to 1024 tokens
6. Batch size increased to 512

The model variants are in a table with columns: Parameters, Layers, d_model, Heads.

### 3.3 How They Describe Training

Training description is extremely sparse — more of a blog post than a conference paper. The paper states:
> "We trained for 1 million steps with a batch size of 512 on WebText."

That is the extent of the training detail in the main paper. No optimizer specification (though it is Adam, inherited from GPT-1), no learning rate schedule (though GPT-1 used cosine decay with warmup), no hardware specification. This is **the least reproducible** of the four papers in terms of training detail.

### 3.4 Prose-to-Equations-to-Tables/Figures Ratio

| Element | Proportion |
|---------|-----------|
| Prose | ~80% |
| Equations | ~0% (no equations in Section 2 at all — the LM objective is described in prose) |
| Figures | ~5% (no architecture figure — the model is a delta from GPT-1) |
| Tables | ~15% (model variant table + WebText statistics) |

GPT-2 has **zero equations** in its Methods section. The language modeling objective is described in one sentence: "At the core of our approach is language modeling."

### 3.5 Length

Section 2 is about **2.5 pages** total. The full paper is only 12 pages including references and appendix.

### 3.6 What Makes This Reproducible

**Poorly reproducible by design.** The paper explicitly states:
> "We were not able to fit the largest model (1.5B parameters) on our existing infrastructure."

This paper was released alongside a staged release strategy (first the small model, then medium, etc.) and was criticized for lack of reproducibility. The code and full WebText dataset were not released.

### 3.7 Signature Writing Techniques

- **"At the core of our approach is X"** — opens the section with a single clear thesis.
- **Data curation emphasis:** The most detailed part of Section 2 is the dataset creation, not the model. This signals that the data is the real contribution.
- **"To do this we only scraped..."** — transparent about data filtering decisions.
- **Model variants presented as a deliberate scaling study**, not as separate architectures: "Our largest model, GPT-2, is a 1.5B parameter Transformer."
- **Minimalist confidence:** Very few hedging terms. Claims are stated directly.

---

## 4. Layer Normalization (Ba, Kiros, Hinton, 2016)

**Section structure:** Sections 2 (Background), 3 (Layer Normalization), 5 (Analysis)

### 4.1 Section Organization

```
2. Background                                        [2 paragraphs + 2 equations]
    2.1 Feed-forward notation
    2.2 Batch normalization explanation

3. Layer Normalization                                [2 paragraphs + 2 equations]
    3.1 Layer Normalized Recurrent Neural Networks    [1 paragraph + equations]

4. Related Work                                       [brief]

5. Analysis                                           [4 subsections]
    5.1 Invariance under Weights and Data Transformations
    5.2 Geometry of Parameter Space During Learning
        5.2.1 Riemannian metric
        5.2.2 The geometry of normalized GLMs
    
6. Experimental Results                               [multiple subsections]
    6.1 Order Embeddings of Images and Language
    6.2 Teaching Machines to Read and Comprehend
    6.3 Skip-Thought Vectors
    6.4 Modeling Binarized MNIST using DRAW
    ...
```

### 4.2 How They Describe Architecture

The Layer Normalization paper uses a **"background first"** structure: Section 2 establishes the notation and the prior art (Batch Normalization) in full detail before Section 3 introduces the new method. This is the standard "methods paper" format.

The key move: **they recycle the exact same notation from Section 2 into Section 3**, making the delta crystal clear. Equation (2) shows BN statistics over the batch; Equation (3) shows LN statistics over the hidden layer. The difference is stated explicitly:

> "The difference between Eq. (2) and Eq. (3) is that under layer normalization, all the hidden units in a layer share the same normalization terms mu and sigma, but different training cases have different normalization terms."

Then Section 3.1 extends LN to RNNs, which is the paper's key application contribution.

### 4.3 How They Describe Training

Training details are embedded in the Experimental Results section (Section 6), not in the methods sections. Each experiment describes its own training setup:
- MNIST: "784-1000-1000-10 fully connected network"
- Skip-thought: "GRU-based encoder-decoder with LN applied to recurrent connections"
- Each experimental subsection states its own hyperparameters.

### 4.4 Prose-to-Equations-to-Tables/Figures Ratio

| Element | Proportion (Sections 2+3 only) |
|---------|-------------------------------|
| Prose | ~60% |
| Equations | ~30% (8+ equations in Sections 2-3 alone) |
| Figures | ~5% |
| Tables | ~5% |

This is the **equation-densest** of the four papers. The methods section reads like a math paper. Examples of equations covered: feed-forward network notation (Eq. 1), BN formulas (Eq. 2), LN formulas (Eq. 3), RNN formulations (Eqs. 4-6 or similar). The analysis section (Section 5) is even more equation-heavy.

### 4.5 Length

The paper is about **10 pages**. Sections 2+3 (Background + Method) are about **3 pages**. The full Analysis section (Section 5) adds another **3 pages**.

### 4.6 What Makes This Reproducible

1. **The method is defined mathematically** — implement any of the equations directly.
2. **Experimental architectures are specified** (layer sizes, activation functions, optimizers).
3. **Comparisons are quantified:** LN vs. BN on specific tasks with specific metrics.
4. **No code release** but the method is simple enough to implement from the equations.

### 4.7 Signature Writing Techniques

- **"We now consider..."** — The most famous opening sentence in the paper, signalling the transition from background to contribution.
- **Compare-and-contrast structure:** Every paragraph about LN is paired with a parallel structure about BN, making the differences jump out.
- **Theoretical justification before empirical results:** Section 5 (Analysis) establishes invariance properties and gradient geometry before Section 6 shows numbers. This makes the paper feel rigorous.
- **Rewriting established equations:** The paper re-states BN formulas fully rather than citing them, so the reader sees the parallel structure. This is a deliberate pedagogical choice.
- **Subsection 3.1 for the domain-specific variant** (RNN) keeps the core method section clean.

---

## 5. Comparative Analysis

### 5.1 Common Patterns Across All Four Papers

**1. Section 3 is always the Methods section.** In all four papers, the method description starts at Section 2 or 3. This is a de facto standard.

**2. One-paragraph-per-component rule.** Every major component gets exactly one paragraph (sometimes two) in the Transformer, BERT, and GPT-2 papers. Nothing gets more than 3 paragraphs except the core contribution.

**3. Hyperparameters are aggregated near the section end.** Either the last paragraph of the subsection or a dedicated training section lists all numerical details together — never scattered throughout the architecture description.

**4. A single architecture table or diagram serves as the orienting device.** The Transformer has Figure 1. BERT has the model size table. GPT-2 has the parameter table. Layer Normalization has the equation comparison. All four give the reader one artifact to anchor on.

**5. Equations are reserved for core innovations only.** The Transformer equations are for attention and positional encoding — what the paper invented. BERT has almost no equations. GPT-2 has none. Layer Normalization is the exception because the method IS an equation.

**6. First-person plural ("we") is dominant.** All four papers use "we" extensively: "We propose," "We use," "We train," "We apply." This is standard in ML but contrasts with chemistry conventions that often prefer passive voice.

**7. Deliberate decisions are explicitly motivated.** Every non-obvious choice gets a "because" clause — the scaling in attention (to avoid small gradients), the 80/10/10 masking (to mitigate the mismatch), the sinusoidal positional encoding (to allow extrapolation).

**8. Ablations are separate from the primary Methods section.** All four papers either put ablations after results (Transformer, BERT) or have a dedicated Analysis section before results (Layer Normalization).

### 5.2 What the Transformer Paper Does Especially Well

The Transformer paper is widely praised for clarity because of several techniques:

**1. The Figure 1 strategy.** The architecture diagram is the most-cited figure in modern ML. Why does it work?
   - It shows the **entire architecture on one page** (encoder + decoder + attention + add&norm + FFN)
   - It uses **consistent visual language** (same box shapes for similar operations)
   - It is **self-contained** — you can understand the architecture without reading the prose
   - It includes **tensor shapes** (the "Nx" annotation, the "d_k = 64" annotation)

**2. The "Why" section (Section 4).** Unlike most papers that bury architectural justification in related work, the Transformer has a standalone analysis section comparing against RNNs and CNNs on computational complexity, parallelism, and path length. This answers "why should I care?" before the results section.

**3. Equation + prose + table triples.** The paper doesn't just give equations — for each component, it provides:
   - A prose explanation (what it does and why)
   - A mathematical formulation
   - A table or comparison (when applicable)
   This triple pattern is extremely effective for clarity.

**4. The learning rate schedule formula.** Instead of saying "we used warmup followed by decay," the paper gives an exact mathematical formula:
   > `lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))`
   This eliminates all ambiguity.

**5. Explicit comparisons in tables.** Table 1 compares self-attention, RNN, and CNN on the same three criteria. This makes the advantage of self-attention immediately visible.

### 5.3 What Makes a Methods Section "Good" vs. "Mediocre"

**Good Methods sections:**

| Feature | Why It Matters |
|---------|----------------|
| One clear diagram showing the full system | Orients the reader; the diagram is what people will cite |
| Every design choice has a "because" | Prevents "why did they do this?" questions |
| One paragraph per component | Readers scan Methods, they don't read every word |
| Hyperparameters in one place | Makes the paper usable as a reference |
| Training details are exact (not vague) | Reproducibility |
| Ablations are cleanly separated | Lets the reader understand the method first, then evaluate it |

**Mediocre Methods sections:**

| Anti-pattern | Why It Fails |
|--------------|-------------|
| Wall-of-text description | No reader will parse it |
| Missing diagram | Readers have to mentally reconstruct the architecture |
| Vague training descriptions ("we used standard settings") | Not reproducible |
| No justification for design choices | Feels like ad-hoc engineering |
| Equations buried in prose without numbering | Hard to reference |
| Ablations mixed into method description | Confuses what the actual method is |

### 5.4 Specific Techniques to Steal for a Computational Chemistry Paper

**From the Transformer:**

1. **ONE big diagram of the full pipeline.** Not separate figures for each component. One figure that shows: input molecule -> representation -> diffusion/VAE/prediction -> output. Include tensor shapes and dimensions in the diagram. Make it the centerpiece of your Methods section.

2. **The triple pattern for each component:** Prose explanation + mathematical formulation + reason. For example:
   - "We represent molecules as graphs G = (V, E) where V are atom features and E are bond features [prose]."
   - "The graph encoder computes h_v^(l+1) = f(W h_v^(l) + sum_{u in N(v)} g(h_v, h_u, e_uv)) [equation]."
   - "This allows the model to capture both atomic and bonding information simultaneously, which is critical for predicting bioisosteric replacements [reason]."

3. **A dedicated "Why This Architecture" section.** Between the Methods and Results, add 2-3 paragraphs explaining why you chose diffusion, or why you use equivariant networks, or why a particular molecular representation works. Use a comparison table if possible.

4. **The exact learning rate formula** instead of a textual description. For computational chemistry papers, consider giving the exact functional form of your learning rate schedule, noise schedule (for diffusion), or annealing schedule.

**From BERT:**

5. **Problem-first structure for each contribution.** Before saying "we use a fragment-based variational autoencoder," first state "a key challenge is representing molecular substructures of variable size while maintaining permutation invariance. To address this, we propose..."

6. **Self-aware limitation statements.** Acknowledge what your method doesn't handle well and explain why the tradeoff is worth it. This increases credibility.

7. **Forward references.** "Full hyperparameters are listed in Appendix A." This keeps the main text clean while providing all details.

8. **Baseline positioning paragraph.** Explicitly state how your method differs from previous work before diving into the method description.

**From GPT-2 (with caution):**

9. **A strong opening thesis sentence.** "At the core of our approach is [one sentence]." This orients the entire section.

10. **Model variants as a deliberate scaling study table.** If you have multiple model sizes, present them as a table with Parameters, Layers, Hidden Dim, Heads (or your domain equivalents). Show that complexity was explored systematically.

**From Layer Normalization:**

11. **Exact equation parallel structure for comparisons.** When comparing to a baseline, use the exact same notation and mathematical structure so the delta is visually obvious. For example, if comparing a new loss function to a standard one, use the same variable names and show the difference in one line.

12. **Theoretical analysis subsection.** Even a simple analysis (invariance properties, computational complexity, gradient properties) makes the paper feel more rigorous. Section 5 of the LN paper is a model for this.

### 5.5 A Recommended Methods Section Structure for a Computational Chemistry Paper

```
3. Methods

3.1 Problem Formulation
    - Notation: molecules as graphs/SMILES, task definition
    - The bioisosteric replacement problem as a conditional generation task
    - Objective function

3.2 Molecular Representation
    - Atom and bond featurization
    - How chemical information is encoded for the model
    - [1 small table of feature dimensions]

3.3 Model Architecture [THE BIG FIGURE]
    - ONE full-pipeline figure
    - Subsection per component:
        3.3.1 Fragment Encoder          [1 paragraph + 1 equation]
        3.3.2 Latent Diffusion Process  [1 paragraph + 1 equation]
        3.3.3 Decoder                   [1 paragraph + 1 equation]
        3.3.4 Equivariance Properties   [1 paragraph]
    - [Why This Architecture subsection - 1 small table comparing to alternatives]

3.4 Training Procedure
    - Optimizer (exact hyperparameters)
    - Learning rate schedule (formula)
    - Noise schedule for diffusion (formula or table)
    - Batch size, gradient clipping, hardware
    - Training time and number of steps

3.5 Data and Preprocessing
    - Dataset sources, sizes, splits
    - Data filtering criteria
    - Training/validation/test splits

3.6 Ablation Study Design
    - What variants we test and why
    - Evaluation metrics
    - [1 summary table of ablation configurations]

3.7 Reproducibility
    - Code availability
    - Random seed handling
    - All hyperparameters in Appendix
```

### 5.6 Writing Checklist

Before submitting, check your Methods section against these criteria:

- [ ] **One big pipeline diagram** — can a reader understand the method from the diagram alone?
- [ ] **One-paragraph-per-component** — no paragraph exceeds ~15 lines
- [ ] **Every equation is numbered** and referenced in the text
- [ ] **Every hyperparameter has an exact value** — no "we used standard settings"
- [ ] **All training details in one place** — not scattered across paragraphs
- [ ] **Each design choice has at least one sentence of motivation** — "we chose X because Y"
- [ ] **Forward references to appendix** for exhaustive details
- [ ] **Comparisons to baselines use the exact same notation** — the delta is visually clear
- [ ] **Ablations are separate from the method** — typically after results or in a dedicated subsection
- [ ] **The first sentence of the Methods section states the core approach** — "At the core of our method is..."
- [ ] **Hardware, training time, and software versions are stated**
- [ ] **Data splits are quantified**
- [ ] **Model variants (if any) are in a clean table** — not described in prose

---

*Compiled from analysis of: Vaswani et al. (2017), "Attention Is All You Need"; Devlin et al. (2019), "BERT: Pre-training of Deep Bidirectional Transformers"; Radford et al. (2019), "Language Models are Unsupervised Multitask Learners"; Ba, Kiros, Hinton (2016), "Layer Normalization."*
