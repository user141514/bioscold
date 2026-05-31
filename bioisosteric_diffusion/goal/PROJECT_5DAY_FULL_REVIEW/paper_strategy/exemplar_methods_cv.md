# Exemplar Methods Sections: Classic Computer Vision Papers

## A Writing-Craft Analysis for Computational Chemistry Method Papers

---

## Table of Contents

1. [ResNet -- He et al., CVPR 2016](#1-resnet)
2. [AlexNet -- Krizhevsky et al., NIPS 2012](#2-alexnet)
3. [VGG -- Simonyan & Zisserman, ICLR 2015](#3-vgg)
4. [DenseNet -- Huang et al., CVPR 2017 (Best Paper)](#4-densenet)
5. [Comparative Analysis](#5-comparative-analysis)
6. [What We Should Steal](#6-what-we-should-steal)

---

## 1. ResNet

**"Deep Residual Learning for Image Recognition"** -- He, Zhang, Ren & Sun, CVPR 2016

### Section Structure

Section 3 ("Deep Residual Learning") is organized as a tight 4-subsection pyramid:

```
3. Deep Residual Learning
  3.1  Residual Learning          (core idea, 1 paragraph)
  3.2  Identity Mapping by Shortcuts  (implementation, 1 paragraph)
  3.3  Network Architectures      (concrete designs, tables + prose)
  3.4  Implementation             (training hyperparameters, 1 page)
```

**Estimated length:** ~3.5 pages of a 9-page paper. The Methods section is approximately 40% of the paper.

### How They Describe the Architecture

**Prose + math + table in precise sequence.**

The paper opens with a problem statement that is both intuitive and mathematically precise:

> "Let us consider H(x) as an underlying mapping to be fit by a few stacked layers (not necessarily the entire net)... We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers."

This is a masterclass in motivation: 1) define notation, 2) state the hypothesis, 3) give the extreme-case intuition.

**Equation (1)** appears immediately: `y = F(x, {Wi}) + x`. Then they explain each piece in prose. They directly state the parameter cost: "The dimensions of x and F must be equal" -- and then address the dimension-mismatch case with two options (A: zero-padding, B: 1x1 projection).

**Table 1** is the architecture table, listing each layer group with output size and filter dimensions. It is the canonical reference. The prose below the table states the *design principles* explicitly as numbered rules:

- "(i) for the same output feature map size, the layers have the same number of filters"
- "(ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer"

**Key structural choice:** They describe the **plain network first**, then say "we simply turn the plain architecture into its residual counterpart." This "baseline then modification" structure makes the innovation instantly legible.

### Training Details (Section 3.4)

One dense paragraph. Every hyperparameter is stated:

> "The image is resized with its shorter side randomly sampled from [256, 480] for scale augmentation [21, 41]. A 224x224 crop is randomly sampled from an image or its horizontal flip... Batch normalization (BN) [16] is performed right after each convolution and before activation... SGD with mini-batch of size 256. The learning rate starts from 0.1 and is divided by 10 when the error plateaus. The models are trained for up to 60 x 10^4 iterations. We use a weight decay of 0.0001 and a momentum of 0.9. We do not use dropout [14]."

**What makes this clear:**
- References prior work for standard augmentations: "following [21, 41]"
- Calls out by name what is DIFFERENT from prior practice: "BN right after each convolution and before activation" (previous BN usage varied)
- Explicitly states what they do NOT use: "We do not use dropout"
- Testing protocol separated: "For testing... 10-crop testing... fully convolutional form... multi-scale evaluation"

### Prose : Equations : Tables Ratio

Approximately **70% prose, 10% equations, 20% tables**. The tables are the architecture specification. Equations are minimal -- only the core residual formula and the projection shortcut formula.

### What Makes It Reproducible

- Every hyperparameter is explicitly listed (batch size, LR schedule, weight decay, momentum, iterations)
- The conversion from plain to residual is so precisely described that a reader can draw both architectures from the text
- The design rules are distilled to 2 numbered principles
- Dimension-mismatch handling is enumerated as concrete options (A, B) with explicit cost comparison
- They say what they did NOT use (dropout), which is as important as what they used

---

## 2. AlexNet

**"ImageNet Classification with Deep Convolutional Neural Networks"** -- Krizhevsky, Sutskever & Hinton, NIPS 2012

### Section Structure

The paper uses a numbered section structure spanning Sections 3-5 for methods:

```
3. The Architecture
  3.1  ReLU Nonlinearity
  3.2  Training on Multiple GPUs
  3.3  Local Response Normalization
  3.4  Overlapping Pooling
  3.5  Overall Architecture
4. Reducing Overfitting
  4.1  Data Augmentation
  4.2  Dropout
5. Details of Learning
```

**Estimated length:** ~5 pages of a 9-page paper. Methods content spans Sections 3, 4, and 5 (roughly 55% of the paper).

**Note on organization:** Unlike ResNet's tight pyramid, AlexNet spreads methods content across three separate sections. "Details of Learning" (Section 5) is surprisingly short -- barely one paragraph of hyperparameters. The bulk of the exposition is in Section 3.

### How They Describe the Architecture

**Subsection 3.5 is the actual architecture description** -- and it reads as a flat numbered list:

> "The first convolutional layer filters the 224x224x3 input image with 96 kernels of size 11x11x3 with a stride of 4 pixels... The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5x5x48..."

**This is purely prose, no equations, no architecture table.** They list each layer sequentially: kernel sizes, strides, padding, output dimensions. In a 9-page paper, the architecture description occupies roughly 1.5 pages of plain text.

The other subsections (3.1-3.4) are organized by **design innovation**, not by architectural component. Each subsection introduces one algorithmic contribution:

- 3.1: ReLU (with Figure 1 showing the 6x speedup over tanh)
- 3.2: Two-GPU model parallelism (with the communication pattern diagrammed)
- 3.3: LRN (with the full equation)
- 3.4: Overlapping pooling

This is a **problem-centric** rather than **architecture-centric** organization: each subsection solves a distinct problem (training speed, GPU memory limits, generalization, overfitting).

### The LRN Equation (Section 3.3)

> b^i_{x,y} = a^i_{x,y} / ( k + alpha * sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)} (a^j_{x,y})^2 )^beta

Parameters explicitly enumerated: k=2, n=5, alpha=10^-4, beta=0.75. This is the only equation in the architecture section.

### Training Details (Section 5)

**Extremely concise -- one dense paragraph:**

> "We used stochastic gradient descent with a batch size of 128, momentum of 0.9, and weight decay of 0.0005... The learning rate was initialized at 0.01 and divided by 10 when the validation error stopped improving... We initialized the weights using a Gaussian distribution with zero mean and standard deviation 0.01. We initialized the biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1."

**What makes this clear:**
- The learning rate schedule is stated: "initialized at 0.01 and divided by 10... it was reduced 3 times"
- The weight initialization distribution is fully specified (Gaussian, mean 0, std 0.01)
- The bias initialization is layer-specific and the rationale is given: "This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs"
- They report hardware: "2 NVIDIA GTX 580 GPUs" and training time: "five to six days"

**What is missing:** Modern reproducibility standards would want exact data augmentation parameters, the random seed policy, and the exact split of kernels across GPUs. But for 2012, this was considered thorough.

### Prose : Equations : Tables Ratio

Approximately **85% prose, 5% equations, 10% figures/tables**. The architecture table that we now associate with AlexNet emerged in blog posts and re-implementations; the original paper describes the full architecture in prose only. The only table of any substance is the ILSVRC-2010/2012 error results in Section 6.

### What Makes It Reproducible

- Despite the prose-only architecture description, each layer's kernel size, stride, padding, and output count are present
- The "7 CNN ensemble" test protocol is clearly described: "Seven CNNs were trained on ILSVRC-2012... their predictions were averaged"
- The bias initialization strategy (constant 1 for certain layers) is unusually specific and helpful
- The Fancy PCA augmentation is described precisely enough (eigenvalues, Gaussian noise sigma) to implement

---

## 3. VGG

**"Very Deep Convolutional Networks for Large-Scale Image Recognition"** -- Simonyan & Zisserman, ICLR 2015

### Section Structure

```
2. ConvNet Configurations
  2.1  Architecture
  2.2  Configurations
  2.3  Discussion
3. Classification Framework
  3.1  Training
  3.2  Testing
  3.3  Implementation Details
```

**Estimated length:** ~4 pages of a 14-page paper (the paper is longer due to extensive experimental tables in Section 4). Methods content is roughly 30% of the total.

**Key organizational insight:** VGG separates the *architecture design* (Section 2) from the *training and evaluation protocol* (Section 3). This is cleaner than AlexNet's three-section sprawl and more modular than ResNet's monolith.

### How They Describe the Architecture

**Section 2.1 is pure architectural specification.** It reads as a compact set of declarations:

> "All convolutional layers use very small 3x3 receptive fields... The convolution stride is fixed to 1 pixel; the spatial padding of 1 pixel is added to the 3x3 convolutional layers to preserve the spatial resolution after convolution. Spatial pooling is carried out by five max-pooling layers... All hidden layers are equipped with the rectification (ReLU) non-linearity."

**Table 1 is the centerpiece.** It lists all six configurations (A-E) in a single table, showing every layer's filter count. This is arguably the most copied table in deep learning literature. The table makes the comparison across depths visually immediate.

**Section 2.3 is the "why" discussion** -- the paper's most influential writing:

> "Rather than using relatively large receptive fields in the first conv layers (e.g. 11x11 with stride 4 in Krizhevsky et al., 2012, or 7x7 with stride 2 in Zeiler & Fergus 2013...), we use very small 3x3 receptive fields throughout the whole net..."

> "So what have we gained by using, for instance, a stack of three 3x3 conv. layers instead of a single 7x7 layer? First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative. Second, we decrease the number of parameters: assuming that both the input and the output of a three-layer 3x3 convolution stack has C channels, the stack is parametrised by 3(3^2 C^2) = 27 C^2 weights; at the same time, a single 7x7 conv. layer would require 7^2 C^2 = 49 C^2 parameters, i.e. 81% more."

**This is the exact format of an ideal architectural justification:**
1. State the design choice ("we use 3x3 filters throughout")
2. Contrast with prior practice (AlexNet's 11x11, Zeiler & Fergus's 7x7)
3. Quantify the benefit (81% fewer parameters for same receptive field)
4. State the secondary benefit (3 ReLUs vs 1 = more discriminative)

### Training Details (Section 3.1 and 3.3)

**Section 3.1 is densely packed with every training detail:**

> "The training was carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent (based on back-propagation [44]) with momentum. The batch size was set to 256, momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to 5x10^-4) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5). The initial learning rate was set to 10^-2, and then decreased by a factor of 10 when the validation set accuracy stopped improving. In total, the learning rate was decreased 3 times..."

The paper then describes the **multi-scale training approach**:

> "Let S be the smallest side of an isotropically-rescaled training image... For single-scale training... S=256... or S=384... For multi-scale training, each training image is individually rescaled by randomly sampling S from a certain range [S_min, S_max] (we used S_min=256 and S_max=512)."

**Section 3.3 describes implementation pragmatics:**

> "The training was performed on a multi-GPU system... The networks were trained on 4 NVIDIA Titan Black GPUs... Training a single network took 2-3 weeks depending on the architecture."

### Prose : Equations : Tables Ratio

Approximately **60% prose, 15% equations, 25% tables**. The large table footprint comes from Table 1 (architectures) and the experimental tables in Section 4. Equations are sparse but used effectively for the parameter-count argument.

### The Multi-GPU Description (Section 3.3)

The paper describes data parallelism succinctly:

> "The training was performed on a multi-GPU system where the mini-batches were split across several GPU..."

This is one sentence. Contrast with AlexNet's three-paragraph description of model parallelism. VGG uses simpler parallelism and describes it proportionally.

### What Makes It Reproducible

- Table 1 is the single reference for all architecture variants
- The ReLU-before-pool or ReLU-after-pool ambiguity is resolved by the table's ordering
- Multi-scale training range [256, 512] is explicitly specified
- Testing protocol is described as "dense evaluation" (fully-convolutional) vs "multi-crop" with a comparison of both
- The "2-3 weeks on 4 Titan Black GPUs" gives an honest resource estimate

---

## 4. DenseNet

**"Densely Connected Convolutional Networks"** -- Huang, Liu, van der Maaten & Weinberger, CVPR 2017 (Best Paper Award)

### Section Structure

```
3. DenseNets
  3.1  Dense Connectivity
  3.2  Composite Function
  3.3  Pooling Layers (Transition Layers)
  3.4  Growth Rate
  3.5  Bottleneck Layers
  3.6  Compression
```

**Estimated length:** ~4 pages of a 12-page paper (Section 3), approximately 33% of the paper.

**Key organizational insight:** DenseNet's Section 3 is unique among the four papers in that it is organized by **hyperparameter**, not by architectural block or training phase. Each subsection introduces and justifies a single knob: the growth rate, the bottleneck factor, the compression factor.

### How They Describe the Architecture

**The paper leads with a single, powerful equation:**

> x_l = H_l([x_0, x_1, ..., x_{l-1}])

where [ ] denotes **concatenation** (not summation, as in ResNet). This one equation is the entire architectural innovation. The rest of the section elaborates on its implications.

**The prose explicitly contrasts with ResNet throughout:**

> "ResNets [12] also connect layer l to layer l-1 via identity connections... Instead of drawing representational power from extremely deep or wide architectures, DenseNets exploit the potential of the network through feature reuse, yielding condensed models that are easy to train and highly parameter-efficient."

**The motif of "collective knowledge" appears repeatedly:**

> "Each layer has access to all preceding feature-maps -- the 'collective knowledge' of the network... Because each layer receives direct supervision from the loss function through the shortcut connections, deep supervision is naturally incorporated."

**The transition layer description is compact:**

> "The layers between contiguous dense blocks are referred to as transition layers and they do the following: batch normalization followed by a 1x1 convolutional layer followed by a 2x2 average pooling layer."

### Training Details

Training details are in Section 4 (Experiments), not Section 3. This differs from ResNet and VGG, which include training details in the method section. For CIFAR:

> "All DenseNet models are trained using SGD with Nesterov momentum [37] and a mini-batch size of 64 for 300 epochs. The initial learning rate is 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs. The weight decay is set to 10^-4 and Nesterov momentum 0.9 without dampening."

For ImageNet:

> "We use SGD with Nesterov momentum and a mini-batch size of 256. The learning rate is set to 0.1 initially, and is divided by 10 at epoch 30 and 60. The weight decay is set to 10^-4, Nesterov momentum to 0.9 without dampening, and we adopt the dropout of 0.2."

**A notable honesty:** The paper discusses a significant memory issue and how they solved it:

> "A naive implementation of dense connectivity... could cause a memory blowup. To overcome this problem... we implement an efficient algorithm that pre-allocates a shared memory buffer."

### Prose : Equations : Tables Ratio

Approximately **75% prose, 10% equations, 15% tables**. Equations are minimal -- the concatenation formula is the only nontrivial one. Architecture specifications are in tables within Section 4 (Experiments), not in the methods section, which is unusual.

### What Makes It Reproducible

- Every hyperparameter is present for both CIFAR and ImageNet
- The growth rate, bottleneck size, and compression factor are each given explicit values
- The memory-efficient implementation is described (shared buffer allocation) -- a practical detail many papers omit
- Dropout rate (0.2) is stated, along with when it is applied (not on the first conv layer)

---

## 5. Comparative Analysis

### 5.1 Common Patterns Across All 4 Papers

**Pattern 1: Problem-First Framing**
Every paper opens its Methods section by stating the problem it solves, not the solution. ResNet opens with the degradation problem, AlexNet with the ReLU speed issue, VGG with the question "how much does depth help?", DenseNet with the vanishing-gradient/information-flow problem. The pattern is universal: **problem before solution**.

**Pattern 2: The Single Central Equation**
Each paper has exactly one equation that captures the core innovation:
- ResNet: y = F(x, {Wi}) + x
- AlexNet: The LRN normalization equation
- VGG: 27C^2 vs 49C^2 parameter count
- DenseNet: x_l = H_l([x0, ..., x_{l-1}])

**Nobody uses more than 2-3 equations** in the entire Methods section. The standard is one equation for the core idea, occasionally one for a design option.

**Pattern 3: Architecture Table as Centerpiece**
Three of four papers (ResNet, VGG, DenseNet for ImageNet) use a formal architecture table. AlexNet is the exception, using prose only. The table is the most-copied artifact from these papers.

**Pattern 4: "Baseline Then Modification"**
When the innovation builds on an existing architecture, the authors describe the baseline first, then the modification. ResNet describes the plain network, then adds shortcuts. VGG describes the generic design (Section 2.1), then gives variants (Section 2.2). DenseNet describes connectivity, then adds bottlenecks and compression.

**Pattern 5: Two-Part Methods Structure (Architecture + Training)**
All four papers separate architectural description from training details. The split is always explicit:
- ResNet: 3.3 (architecture) vs 3.4 (implementation)
- AlexNet: Section 3 (architecture) vs Section 5 (learning details)
- VGG: Section 2 (configurations) vs Section 3 (classification framework)
- DenseNet: Section 3 (architecture) vs Section 4 (experiments, which includes training)

**Pattern 6: Cost Transparency**
Every paper reports parameter counts or FLOPs. This is treated as a table row or an explicit sentence. The cost of the innovation is always disclosed alongside its benefit.

**Pattern 7: Negative Results Included**
The strongest papers explicitly report what did NOT work. ResNet mentions that "if F has only a single layer, the equation reduces to a linear layer, and we have not observed advantages." VGG reports "LRN does not improve performance." DenseNet acknowledges that full dense connectivity causes memory issues. This honesty increases credibility.

### 5.2 Differences in Style

**Narrative density:**
- **AlexNet** is the most narrative (85% prose). It reads like a story: "We encountered X problem, we tried Y, and here's what happened." This reflects the 2012 style of exploratory research reporting.
- **VGG** is the most declarative and compact. "We use this. We do not use that. The parameters are these." It reads like a design specification.
- **ResNet** is the most pedagogical. It opens with a conceptual argument, formalizes it mathematically, then gives concrete designs. It explicitly teaches the reader why the idea works.
- **DenseNet** is the most modular. Each subsection is a self-contained mini-essay on one design knob. This makes it easy to reference, but harder to read from start to finish.

**Table usage:**
- VGG has the most effective single table (Table 1 shows all configurations at a glance)
- ResNet has the most complete table (Table 1 in the paper covers the full architecture)
- AlexNet has no architecture table (the weakest for reproducibility)
- DenseNet puts its ImageNet architecture tables in the Experiments section, not Methods -- an unusual choice

**Training detail location:**
- ResNet and VGG embed training details in the Methods section
- DenseNet puts them in the Experiments section
- AlexNet has a separate "Details of Learning" section
- **Recommendation:** Embed training details in or immediately after architecture description. This is the modern standard, and readers expect it there.

**Motivation style:**
- VGG is the strongest at justifying design choices ("81% more parameters... three ReLUs instead of one")
- DenseNet is the strongest at connecting design choices to a core theme (information flow)
- ResNet is the strongest at intuitive analogies ("it is easier to push the residual to zero")
- AlexNet is the weakest at explicit motivation -- many choices are presented as empirical observations rather than principled designs

### 5.3 What Makes a Methods Section "Good" vs "Mediocre"

**A good Methods section:**

| Quality | How the exemplars achieve it |
|---------|-------------------------------|
| **Legibility** | One equation per idea. Distill design rules to numbered lists. |
| **Completeness** | Every hyperparameter stated explicitly. No reliance on "standard settings" without reference. |
| **Comparability** | Report parameter counts and FLOPs. Baseline the approach against a known reference. |
| **Honesty** | Report what did NOT work. State caveats. |
| **Modularity** | Each subsection is self-contained. Architecture vs training vs evaluation are clearly separated. |
| **Motivation** | Every design choice is justified. "We use X because Y." |
| **Cost transparency** | "We do not use dropout" is as important as what you use. "No extra parameters" is stated explicitly. |
| **Readability hierarchy** | Section title tells you what the subsection contains. Tables summarize what prose explains. |

**A mediocre Methods section:**
- Describes architecture in prose only (AlexNet model)
- Mixes training details and architecture description in the same paragraph
- Uses "We employed a convolutional neural network" without specifying filter sizes or strides
- Omits hyperparameters or says "learning rate was chosen by cross-validation" without reporting the chosen value
- Does not report the hardware or training time (this matters for reproducibility assessment)
- Has no visual summary (table or figure) of the architecture

### 5.4 Methods Section Length Comparison

| Paper | Methods pages | Total pages | Methods % | Subsections |
|-------|--------------|-------------|-----------|-------------|
| ResNet | ~3.5 | 9 | 40% | 4 |
| AlexNet | ~5 | 9 | 55% | 9 (across 3 sections) |
| VGG | ~4 | 14 | 30% | 6 (across 2 sections) |
| DenseNet | ~4 | 12 | 33% | 6 |

**Takeaway:** 30-40% of total paper length is the sweet spot. AlexNet is unusually high because the Methods section also includes what modern papers would put in Results. The modern trend (ResNet, DenseNet) is a tighter Methods section with a separate, longer Experiments section.

---

## 6. What We Should Steal for Our Computational Chemistry Method Paper

### 6.1 Structure Template (from VGG + ResNet)

Adopt the two-part structure:

```
Section 3: Method
  3.1  Problem Formulation        (equation + prose)
  3.2  Model Architecture          (table + diagram)
  3.3  Training Procedure          (hyperparameter table)
  3.4  Evaluation Protocol         (metrics, baselines, datasets)
Section 4: Experiments
  4.1  Dataset Descriptions
  4.2  Baselines and Comparisons
  4.3  Ablation Studies
  4.4  Case Studies
```

### 6.2 The "One Equation" Rule (from all four papers)

Identify the **single equation** that captures your method's core innovation. For us, this might be the equivariant diffusion loss, the fragment-conditioned score matching equation, or the isostere replacement likelihood. Put it front and center, identify each variable, state what it means physically. Do not bury it in a sea of notation.

### 6.3 The Architecture Table (from VGG Table 1)

VGG's Table 1 is the gold standard: it shows all architecture variants on one page, making comparison immediate. For a computational chemistry paper, this could be a table showing:
- Each layer block: name, input dimension, output dimension, number of parameters
- Fragment encoder: atom types, bond types, embedding dimension
- Diffusion model: time embedding, number of layers, hidden dimension
- Equivalent to VGG's "conv3-64, conv3-128..." notation but for molecular features

### 6.4 The "Baseline Then Modification" Frame (from ResNet)

If our method builds on an existing framework (diffusion models, equivariant GNNs), describe the baseline model first, then show what we changed. This makes the novelty instantly visible. Use a sentence like: "The baseline follows [ref]. Our modification replaces the standard denoising objective with a fragment-conditional score matching loss..."

### 6.5 Design Principles as Numbered Lists (from ResNet)

ResNet's two design rules ("(i) for the same output feature map size... (ii) if the feature map size is halved...") are the most frequently quoted sentences in the paper. We should similarly distill our architectural choices to 2-4 numbered principles.

### 6.6 The Parameter-Cost Comparison (from VGG Section 2.3)

VGG's "27C^2 vs 49C^2 = 81% more" is devastatingly effective. We should identify a comparable comparison: "Our fragment-conditional model uses M parameters vs the unconditioned baseline's N parameters, a X% reduction." Put the raw numbers AND the percentage.

### 6.7 Cost Transparency (from ResNet + VGG)

Explicitly state:
- "This introduces no additional parameters" (if true)
- "This adds N parameters, representing a X% increase"
- Training time, hardware, and wall-clock comparison

### 6.8 The Honest Implementation Note (from DenseNet)

DenseNet's admission that "a naive implementation could cause a memory blowup" and their description of the shared-memory buffer fix is exactly the kind of practical detail that separates reproducible papers from vague ones. We should include our own implementation challenge and how we solved it.

### 6.9 Concrete Quotes to Emulate

| Citation | Quote | What makes it effective |
|----------|-------|------------------------|
| ResNet 3.1 | "If an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers." | Extreme-case intuition; makes the abstract concrete |
| VGG 2.3 | "81% more parameters... three non-linear rectification layers instead of one" | Numbers AND a secondary benefit, not just one claim |
| DenseNet 3.4 | "Each layer has access to all preceding feature-maps -- the 'collective knowledge' of the network" | A memorable phrase that encodes the entire design philosophy |
| AlexNet 4.1 | "This reduces the top-1 error by over 1 percent" | Each design choice is tied to a quantitative improvement |
| ResNet 3.2 | "The shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers. Identity shortcuts introduce neither extra parameter nor computational complexity." | The no-cost framing is repeated like a mantra |

### 6.10 What to Avoid

From the weaknesses observed:

1. **Do not describe the full architecture in prose only** (AlexNet's mistake). Always provide a table.

2. **Do not mix training details and architecture description in the same paragraph.** Separate them into distinct subsections.

3. **Do not omit hyperparameters.** Readers should be able to reproduce training from the paper alone. Every missing number creates uncertainty.

4. **Do not leave "design choice" unmotivated.** Every architecture decision should have a "because" attached.

5. **Do not skip the negative results.** ResNet's zero-padding vs projection analysis (Option A vs B vs C) is the model of thoroughness. Show that you tested alternatives.

6. **Do not conflate the contribution with the baseline.** If using diffusion models, clearly separate what is the standard diffusion framework and what is our innovation. ResNet's "plain network first, then residual" is the pattern.

---

## References

- ResNet: He et al., "Deep Residual Learning for Image Recognition," CVPR 2016. arXiv:1512.03385
- AlexNet: Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks," NIPS 2012.
- VGG: Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015. arXiv:1409.1556
- DenseNet: Huang et al., "Densely Connected Convolutional Networks," CVPR 2017 (Best Paper). arXiv:1608.06993
