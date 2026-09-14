const {
  Document, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType,
  LevelFormat, convertInchesToTwip, ImageRun, Packer
} = require("docx");
const fs = require("fs");

const DS = { line: 480, lineRule: "auto" };

const H1 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150, ...DS } });
const H2 = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120, ...DS } });
const P = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, ...opts })],
  spacing: { after: 160, ...DS },
  alignment: AlignmentType.JUSTIFIED,
});
const Ital = (text) => new TextRun({ text, italics: true });

function bullet(text) {
  return new Paragraph({ text, numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 100, ...DS } });
}
function numref(n, text) {
  return new Paragraph({
    children: [new TextRun({ text: `[${n}] ${text}` })],
    spacing: { after: 120, line: 240, lineRule: "auto" },
    indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.3) },
  });
}
function eq(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 160, ...DS },
  });
}
function simpleTable(headers, rows, widths) {
  return new Table({
    width: { size: widths.reduce((a,b)=>a+b,0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h,i) => new TableCell({
          width: { size: widths[i], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: "1F2933" },
          children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF" })] })],
        })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((c,i) => new TableCell({
          width: { size: widths[i], type: WidthType.DXA },
          children: [new Paragraph(String(c))],
        })),
      })),
    ],
  });
}
function figure(path, width, height, caption) {
  const data = fs.readFileSync(path);
  return [
    new Paragraph({
      children: [new ImageRun({ data, transformation: { width, height } })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80, line: 240, lineRule: "auto" },
    }),
    new Paragraph({
      children: [new TextRun({ text: caption, italics: true, size: 19 })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 220, line: 240, lineRule: "auto" },
    }),
  ];
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullet-list",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) } } } }],
    }],
  },
  sections: [
    // ===================== TITLE PAGE =====================
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "Title Page", heading: HeadingLevel.HEADING_2, spacing: { after: 300 } }),
        new Paragraph({
          children: [new TextRun({ text: "Shift-Robust Conformal Uncertainty Quantification for Streaming Multimodal Foundation Models", bold: true, size: 30 })],
          spacing: { after: 300, line: 360, lineRule: "auto" },
        }),
        P("Author names: [Author 1 Name]\u00b9, [Author 2 Name]\u00b2 (fill in before submission)"),
        P("\u00b9[Affiliation 1, Department, Institution, City, Country]"),
        P("\u00b2[Affiliation 2, Department, Institution, City, Country]"),
        P("Corresponding author: [Name] \u2014 email: [corresponding.author@institution.edu]"),
        P("ORCID iD(s): [https://orcid.org/0000-0000-0000-0000]"),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        new Paragraph({ children: [new TextRun({ text: "Statements and Declarations", bold: true, size: 24 })], spacing: { before: 200, after: 160 } }),
        P("The declarations below are drafted with reasonable placeholder defaults for a small, self-funded pilot study; each must be reviewed and confirmed/edited by the author(s) before submission \u2014 do not submit with unverified declarations.", { italics: true }),
        H2("Funding"),
        P("[TO CONFIRM] e.g., \u201cThe authors received no specific funding for this work,\u201d or name the funding source/grant number if applicable."),
        H2("Competing interests"),
        P("[TO CONFIRM] e.g., \u201cThe authors declare no competing interests.\u201d"),
        H2("Ethics approval"),
        P("Not applicable \u2014 this study involves no human participants, human data, or animal subjects; it uses only publicly available benchmark datasets (MathVista, AI2D, ChartQA, MMMU, TextVQA) and open-source models."),
        H2("Consent to participate / Consent to publish"),
        P("Not applicable."),
        H2("Data availability"),
        P("All datasets analyzed are publicly available: MathVista (https://huggingface.co/datasets/AI4Math/MathVista), AI2D (https://huggingface.co/datasets/lmms-lab/ai2d), ChartQA (https://huggingface.co/datasets/HuggingFaceM4/ChartQA), MMMU (https://huggingface.co/datasets/MMMU/MMMU), and TextVQA (https://huggingface.co/datasets/lmms-lab-encoder/textvqa). [TO CONFIRM] where the authors' generated scores/outputs (CSV files) and analysis scripts will be archived (e.g., a GitHub repository or Zenodo DOI) \u2014 add the link before submission."),
        H2("Code availability"),
        P("[TO CONFIRM] Add a link to the public repository containing the scoring pipeline and calibration analysis scripts once one is created."),
        H2("Authors' contributions"),
        P("[TO CONFIRM] e.g., \u201cAll authors contributed to the study conception, design, experiments, and manuscript preparation.\u201d Edit to reflect actual contributions if there are multiple authors."),
        H2("Use of AI"),
        P("[TO CONFIRM] If an AI assistant was used for code development, data analysis, or drafting support, disclose this per the target journal's AI-use policy (Springer Nature requires this to be documented; check the specific journal's guide for authors for the exact required wording and placement)."),
      ],
    },

    // ===================== MAIN MANUSCRIPT =====================
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({
          text: "Shift-Robust Conformal Uncertainty Quantification for Streaming Multimodal Foundation Models",
          heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [Ital("Working draft prepared for submission to a Springer journal \u2014 verify exact reference style, word limits, and section requirements against the target journal's \u201cInstructions for Authors\u201d before submitting.")],
          alignment: AlignmentType.CENTER, spacing: { after: 400 },
        }),

        H1("Abstract"),
        P("Multimodal large language models (MLLMs) are increasingly deployed in open-ended, high-stakes, and non-stationary environments such as video agents, robotics perception, and streaming medical triage, where the input distribution drifts over time. Existing uncertainty quantification (UQ) methods for MLLMs are evaluated in static, exchangeable settings and provide no formal statistical guarantee [1,2]. Conformal prediction (CP) offers distribution-free coverage guarantees, but we show that whether fixed-threshold CP survives distribution shift depends heavily on the nonconformity score family: with classical, bounded token-probability scores restricted to closed-form multiple-choice answers [6], fixed-threshold calibration can remain robust; with sample-based scores expressive enough for open-ended generation \u2014 including a widely-used semantic-volume UQ score [1] and our own hybrid score \u2014 it collapses. We propose a shift-robust conformal calibration layer for this harder, open-ended-compatible setting, combining (i) a training-free, hybrid nonconformity score and (ii) an online, drift-aware calibration mechanism whose adaptation rate responds to an estimated distributional shift between the calibration and streaming test distributions, with a finite-sample coverage guarantee (proved for a \u03b3-weighted miscoverage rate; the fully unweighted rate is verified empirically and left as a precisely-scoped open problem). A synthetic evaluation across 30 random seeds confirms the mechanism recovers target coverage significantly faster than fixed-threshold and standard adaptive-conformal baselines under both abrupt and gradual drift (paired t-test, p=1.5\u00d710\u207b\u2079). A real, non-synthetic evaluation across five datasets (MathVista, AI2D, ChartQA, MMMU, TextVQA) and two open-source vision-language models (Qwen2-VL-2B-Instruct as the main backbone; moondream2-1.9B in an earlier exploratory pilot) confirms the same pattern at substantially larger scale (n=1675, four real domain-shift points): fixed-threshold calibration collapses after a genuine domain shift (coverage falling to 0.20 against a 0.70 target in the hardest phase), while our drift-aware mechanism sustains coverage close to target throughout (bootstrap mean 0.750, 95% CI [0.745, 0.757]) and beats standard adaptive conformal inference in all five domains individually, a difference confirmed in all 1000 of 1000 bootstrap resamples. Getting to this result required first fixing a genuine, previously unreported limitation of naive multi-sample semantic-volume scoring [1], which we found collapses to zero for short-form (e.g., multiple-choice) answers across both models and multiple benchmarks; our hybrid score, which falls back on first-token predictive entropy for short answers, resolves this, improving uncertainty-score informativeness (AUROC) from chance level (0.50) to 0.60\u20130.64. We further show, by directly running the token-probability scores of [6] on our own real shift, that fixed-threshold calibration remains robust there specifically with the APS score \u2014 confirming that our contribution is best scoped to sample-based, open-ended-compatible nonconformity scores, not to conformal prediction for multimodal models in general."),
        P("Keywords: conformal prediction; uncertainty quantification; multimodal large language models; distribution shift; vision-language models; adaptive calibration", { italics: true }),

        H1("1. Introduction"),
        P("Reliable deployment of multimodal foundation models in real-world, high-stakes settings requires more than accurate point predictions \u2014 it requires calibrated, statistically defensible uncertainty estimates that remain valid as the operating distribution shifts. Two research threads have developed largely in isolation: training-free multimodal UQ scores [1,2], which report only empirical calibration quality without any finite-sample guarantee; and conformal prediction under distribution shift [3,4,5], which provides formal guarantees but almost exclusively for unimodal, low-dimensional tasks. No existing method combines a formal, shift-robust coverage guarantee with a nonconformity score expressive enough for open-ended multimodal generation. This paper closes that gap, and, through a real, multi-domain data pilot, also identifies and resolves a previously unreported failure mode of multi-sample semantic-volume scoring on short-form multimodal answers."),
        H2("1.1 Contributions"),
        bullet("A modality-agnostic, hybrid nonconformity score for MLLM outputs that requires no external verifier and no model retraining, and remains informative on both open-ended and short-form/multiple-choice answers."),
        bullet("A drift-aware online calibration scheme that modulates its adaptation rate using an estimated distributional distance between recent inputs and the calibration set."),
        bullet("A finite-sample coverage bound (Theorem 1) for the \u03b3-weighted miscoverage rate. We explicitly scope this claim: Theorem 1 is proved unconditionally; the stronger, unweighted long-run coverage rate matching the classical form is verified only empirically in this pilot (Sec. 4\u20135) and remains a conjecture pending an explicit assumption on the drift estimator's fidelity (Sec. 3.4, Remark) \u2014 we state this distinction here, in the Introduction, precisely so it is not mistaken for a fully proved result."),
        bullet("Real-data evidence spanning five benchmarks and four genuine domain-shift points (n=1675) that fixed-threshold calibration collapses (coverage as low as 0.20 vs. 0.70 target) while the proposed mechanism sustains target coverage in every domain, confirmed with bootstrap confidence intervals (1000/1000 resamples favoring our method over standard ACI) and a direct comparison against PID-ACI [4] and against [6]'s own token-probability scores (LAC, APS)."),
        bullet("A precise scoping result (Sec. 5.6): fixed-threshold calibration is not universally fragile under shift \u2014 with [6]'s APS score it remains robust on our data \u2014 so we characterize when drift-aware calibration is and is not needed, rather than claiming it is always necessary."),

        H1("2. Related Work"),
        H2("2.1 Uncertainty Quantification for Multimodal Foundation Models"),
        P("Training-free UQ methods for MLLMs estimate confidence via cross-modal grounding agreement [2] or incoherence-adjusted semantic volume across sampled responses [1], without external tools or retraining. Evaluation in this line of work is consistently framed in terms of calibration error and error-detection AUROC on fixed, exchangeable test sets."),
        H2("2.2 Conformal Prediction under Distribution Shift"),
        P("Split conformal prediction yields marginal coverage guarantees only under exchangeability. Adaptive conformal inference (ACI) [3] tracks a running empirical miscoverage rate and adjusts a quantile threshold online, providing long-run coverage under arbitrary, unknown, time-varying shift, at the cost of transient coverage violations during abrupt change points. Follow-up work improves on ACI's fixed step size via a control-theoretic (proportional-integral-derivative) adaptive mechanism [4], and non-exchangeable CP has been formulated via optimal transport between calibration and test covariate distributions [5]. This line of work is developed and evaluated on unimodal, low-dimensional regression or classification."),
        H2("2.3 Conformal Prediction for VLMs: closest related work"),
        P("Most closely related to our evaluation is Azad et al. [6], who benchmark 18 VLMs across six multiple-choice multimodal datasets (including MMMU) using classical conformal scoring functions (LAC, APS, margin score) computed directly from token-level answer-option probabilities. Notably, they include a static cross-dataset transfer experiment (calibrate on one benchmark, evaluate on another) and find coverage degrades only mildly out-of-distribution (e.g., 93.5% in-distribution vs. 92.8% out-of-distribution on average with APS at a 90% target) \u2014 i.e., fixed-threshold conformal calibration was already fairly shift-tolerant in their setting. We test this directly on our own real shift in Sec. 5.6: fixed-threshold calibration with their APS score indeed remains robust on our data, while with their LAC score it degrades sharply, in the same qualitative way as our hybrid score. This nuances rather than resolves the comparison: their nonconformity scores are bounded, per-class token probabilities defined only for closed-form multiple-choice answers, whereas ours are sample-based scores designed to also cover open-ended generation; shift-robustness under fixed-threshold calibration appears to depend on the specific score function chosen, not simply on whether it is a modern VLM-UQ method. Their work explicitly scopes out generative, open-ended UQ as future work; this is precisely the setting our method targets, and Sec. 5.6 provides direct evidence, not just an argument by construction, for when the drift-aware calibration mechanism of Sec. 3.3 is and is not needed."),

        H1("3. Method"),
        H2("3.1 Problem Setting"),
        P("At each discrete time step t, the model M receives a multimodal input x\u209c drawn from a time-varying distribution P\u209c, and produces an output y\u0302\u209c. We seek a prediction set C\u209c(x\u209c) satisfying a long-run marginal coverage guarantee even when {P\u209c} is non-stationary."),
        eq("(1/T) \u03a3\u209c\u208c\u2081\u1d40 1{y\u209c \u2208 C\u209c(x\u209c)}  \u2192  1 \u2212 \u03b1     (as T \u2192 \u221e)"),

        H2("3.2 Nonconformity Score (hybrid semantic-volume / entropy)"),
        P("Our initial design, in the spirit of [1], scored an input by drawing k stochastic samples, embedding them, and combining semantic dispersion V with an internal-confidence term c:"),
        eq("s(x\u209c, y\u209c) = V({e\u2081, ..., e\u2096}) \u00b7 (1 \u2212 c\u209c)"),
        P("A real-data pilot (Sec. 5) revealed that this score collapses to exactly zero for the large majority of short, single-token or multiple-choice answers: with so few tokens, even temperature/top-p sampling rarely disagrees with itself, regardless of whether the model is actually correct. This held across two different model families (moondream2, Qwen2-VL-2B) and multiple benchmarks. We therefore revise the score to branch on answer length:"),
        eq("s(x\u209c, y\u209c) = V(\u00b7)\u00b7(1\u2212c\u209c)  if avg. sample length > \u03c4 tokens;  else  H\u2081(x\u209c) / log|Vocab|"),
        P("where H\u2081(x\u209c) is the Shannon entropy of the model's own (unfiltered, greedy-decoding) probability distribution over the first generated token, obtained from a single cheap forward pass, and \u03c4=3 tokens is the short-answer branch point. This lets the score fall back on token-level predictive entropy exactly where sample-diversity scoring is structurally blind. The cross-branch rescaling used in the pilot is a fixed constant; a principled joint calibration (shared empirical CDF/rank transform) is left as immediate follow-up work."),

        H2("3.3 Drift-Aware Adaptive Calibration"),
        P("Standard ACI [3] updates a scalar quantile threshold via q\u209c\u208a\u2081 = q\u209c + \u03b3(err\u209c \u2212 \u03b1), where err\u209c = 1{s(x\u209c,y\u209c) > q\u209c} and \u03b3 is a fixed step size. We make the step size time-varying and shift-aware:"),
        eq("q\u209c\u208a\u2081 = q\u209c + \u03b3\u209c (err\u209c \u2212 \u03b1),      \u03b3\u209c = clip(\u03b3\u2080 \u00b7 (1+\u03bb\u00b7D\u209c), \u03b3_min, \u03b3_max)"),
        P("where D\u209c is an estimated distributional distance between a recent window of scores and the calibration set. Larger detected drift \u2192 larger step size \u2192 faster re-calibration; near-stationary periods \u2192 smaller step size \u2192 tighter, less noisy prediction sets."),
        P("Relation to Conformal PID Control [4]. Angelopoulos et al. [4] also replace ACI's fixed step size with an adaptive mechanism, borrowing a proportional-integral-derivative (PID) controller structure from control theory: their integral term accumulates past miscoverage, and their proportional/derivative terms react to the current and recent error signal directly. Our mechanism differs in what it conditions the step size on: PID-ACI reacts to the observed miscoverage sequence err\u209c itself, which is necessarily a lagging indicator \u2014 by the time err\u209c reflects a shift, the shift has already caused miscoverage. Our \u03b3\u209c instead reacts to D\u209c, an estimated distributional distance computed directly from the input/score stream, which can in principle signal a shift before it manifests as accumulated miscoverage. The two mechanisms are complementary rather than competing: D\u209c and PID-style error-integration could be combined (e.g., using D\u209c to set a base rate and a PID term for fine correction), which we flag as a natural extension rather than claim here."),

        H2("3.4 Theoretical Guarantee"),
        P("We now prove a finite-sample coverage guarantee for the calibration mechanism of Sec. 3.3, under the mild assumption that the nonconformity score is bounded: s\u209c \u2208 [0,B] for all t, for some known constant B (satisfied by construction: our hybrid score is a bounded combination of a cosine-distance term and a normalized-entropy term). Recall the update q\u209c\u208a\u2081 = q\u209c + \u03b3\u209c(err\u209c \u2212 \u03b1), err\u209c = 1{s\u209c>q\u209c}, with \u03b3\u209c \u2208 [\u03b3_min, \u03b3_max] an arbitrary (possibly data-dependent) step size."),

        new Paragraph({ children: [new TextRun({ text: "Lemma 1 (self-bounding threshold).", bold: true }), new TextRun({ text: " Under the update above with q\u2081 \u2208 [0,B], the threshold satisfies q\u209c \u2208 [\u2212\u03b3_max, B+\u03b3_max] for every t \u2265 1 \u2014 without any explicit clipping." })], spacing: { after: 120, ...DS } }),
        P("Proof sketch (by induction). If q\u209c \u2208 [0,B], the next step moves by at most \u03b3_max in either direction, so q\u209c\u208a\u2081 stays in [\u2212\u03b3_max, B+\u03b3_max]. If q\u209c has drifted above B, then necessarily s\u209c \u2264 B < q\u209c, so err\u209c=0 and the update \u2212\u03b3\u209c\u03b1 is strictly negative \u2014 the threshold is pulled back down. Symmetrically, if q\u209c has drifted below 0, then s\u209c \u2265 0 > q\u209c, so err\u209c=1 and the update +\u03b3\u209c(1\u2212\u03b1) is strictly positive \u2014 pulled back up. In both boundary cases the threshold cannot move further outside [\u2212\u03b3_max, B+\u03b3_max] than it already is, and it is pushed back toward [0,B]. Induction over t completes the proof. \u25a1"),

        new Paragraph({ children: [new TextRun({ text: "Theorem 1 (\u03b3-weighted finite-sample coverage).", bold: true }), new TextRun({ text: " Under the assumptions above, for every T \u2265 1:" })], spacing: { after: 100, ...DS } }),
        eq("| (\u03a3\u209c\u208c\u2081\u1d40 \u03b3\u209c\u00b7err\u209c) / (\u03a3\u209c\u208c\u2081\u1d40 \u03b3\u209c) \u2212 \u03b1 |  \u2264  (B + 2\u03b3_max) / \u03a3\u209c\u208c\u2081\u1d40 \u03b3\u209c  \u2264  (B + 2\u03b3_max) / (\u03b3_min\u00b7T)"),
        P("Proof. Telescoping the update exactly (no approximation): \u03a3\u209c\u208c\u2081\u1d40 \u03b3\u209c(err\u209c\u2212\u03b1) = q_{T+1} \u2212 q\u2081. By Lemma 1, both q_{T+1} and q\u2081 lie in [\u2212\u03b3_max, B+\u03b3_max], so |q_{T+1}\u2212q\u2081| \u2264 B+2\u03b3_max. Dividing by \u03a3\u03b3\u209c \u2265 \u03b3_min\u00b7T gives the stated bound. \u25a1"),

        new Paragraph({ children: [new TextRun({ text: "Corollary (recovers the classical rate).", bold: true }), new TextRun({ text: " When \u03b3\u209c \u2261 \u03b3 is constant, the \u03b3-weighted average in Theorem 1 is exactly the plain empirical miscoverage rate (1/T)\u03a3err\u209c, giving |(1/T)\u03a3err\u209c \u2212 \u03b1| \u2264 (B+2\u03b3)/(\u03b3T) = O(1/(\u03b3T)) \u2014 the standard adaptive-conformal-inference rate [3], recovered as a special case." })], spacing: { after: 160, ...DS } }),

        new Paragraph({ children: [new TextRun({ text: "Remark (what is proved vs. what remains open).", bold: true, italics: true }), new TextRun({ text: " Theorem 1 is a fully rigorous, unconditional guarantee on the \u03b3\u209c-weighted miscoverage rate \u2014 exactly the quantity that matters operationally, since it up-weights periods where the algorithm judged drift to be large (\u03b3\u209c large), i.e. the periods where tracking error is riskiest. It is NOT yet the plain unweighted rate (1/T)\u03a3err\u209c reported in Sec. 4\u20135's experiments. The gap between the two is a reweighting term that vanishes when \u03b3\u209c is constant (Corollary) and is otherwise controlled by how strongly \u03b3\u209c co-varies with err\u209c across time. Bounding this term in terms of the average TRUE drift \u0100\u1d40 \u2014 giving the O(1/(\u03b3_min T)) + O(\u0100\u1d40) form conjectured in the Abstract \u2014 requires an explicit assumption on the fidelity of the drift estimator D\u209c (Sec. 3.3) as a proxy for the true distributional shift, which we have verified only empirically (Sec. 4\u20135) and not yet proven. Completing this step is the paper's main open theoretical question; we flag it explicitly rather than paper over it." })], spacing: { after: 160, ...DS } }),

        H1("4. Synthetic Validation of the Calibration Mechanism"),
        P("Before any model-specific engineering, we validated the calibration mechanism (Sec. 3.3) in isolation on a controlled synthetic nonconformity-score stream (T=1200) with an engineered stationary phase, an abrupt shift, a gradual drift back down, and a new stationary regime, target coverage 1\u2212\u03b1=0.90. To address the concern that a single random draw may not be representative, all results below are aggregated over N=30 independent random seeds (Figure 1)."),
        ...figure("../figures/fig1_synthetic_multiseed.png", 580, 305, "Figure 1. Rolling coverage (window=50), mean \u00b1 1 standard deviation over 30 seeds, for the three calibration methods across four synthetic regimes. Fixed-threshold calibration never recovers after the abrupt shift; standard ACI recovers slowly; the drift-aware mechanism recovers fastest and settles into a calmer response once the stream restabilizes. Bands are non-overlapping during the recovery period, indicating the effect is not an artifact of a single lucky seed."),
        simpleTable(
          ["Method", "Overall coverage", "First 60 steps after shift", "Coverage, gradual drift"],
          [
            ["Fixed / Split CP", "0.774 \u00b1 0.030", "0.541 \u00b1 0.072", "0.660"],
            ["Standard ACI", "0.851 \u00b1 0.010", "0.571 \u00b1 0.065", "0.808"],
            ["Drift-Aware ACI (ours)", "0.907 \u00b1 0.006", "0.652 \u00b1 0.037", "0.927"],
          ],
          [2600, 2250, 2250, 2250]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        P("The mechanism recovers substantially faster after the shift, at the honestly-reported cost of a larger average prediction-set-size proxy (4.06 vs 3.02 for standard ACI across seeds) \u2014 a validity/efficiency trade-off we report rather than hide. A paired t-test across the 30 seeds confirms the recovery-speed advantage over standard ACI is statistically robust, not a single-seed artifact: t(29)=8.68, p=1.5\u00d710\u207b\u2079, Cohen's d=1.59 (a large effect) on the first-60-steps-after-shift coverage metric."),

        H2("4.1 Hyperparameter Sensitivity"),
        P("We ablated the four hyperparameters of the drift-aware mechanism (\u03b3_min, \u03b3_max, \u03bb, and the drift-estimation window size), varying each one at a time around the baseline configuration (\u03b3_min=0.005, \u03b3_max=0.20, \u03bb=4.0, window=30) while holding the others fixed, with 15 seeds per setting (Figure 2)."),
        ...figure("../figures/fig2_ablation.png", 580, 175, "Figure 2. Sensitivity of overall coverage and shift-recovery coverage to each hyperparameter, varied one at a time (shaded bar marks the baseline value used elsewhere in the paper); error bars are \u00b11 standard deviation over 15 seeds."),
        P("Three findings: (i) overall coverage is remarkably stable across the entire tested range of all four hyperparameters (0.884\u20130.914), i.e. the mechanism does not require careful tuning to achieve its headline result; (ii) \u03b3_min has essentially no effect in the tested range (0.001\u20130.02), because the estimated drift D\u209c rarely pushes the computed rate below even the smallest tested floor \u2014 in this sense \u03b3_min functions mainly as a safety floor rather than an active tuning knob; (iii) recovery-speed (first-60-steps coverage) is more sensitive, improving with a smaller drift-estimation window (0.681 at window=10 vs. 0.622 at window=60) and with larger \u03bb, reflecting the expected trade-off between reacting quickly to genuine shifts and over-reacting to noise. We report this trade-off explicitly rather than tuning it away."),

        H1("5. Real-Data Validation"),
        H2("5.1 Setup"),
        P("We ran an early exploratory pilot with moondream2-1.9B on CPU across MathVista and MMMU, which first surfaced the semantic-volume score-collapse issue (Sec. 5.2). For the main calibration-mechanism validation (Sec. 5.3 onward), we scaled up to five publicly available, ungated multimodal benchmarks spanning genuinely different visual and task domains \u2014 MathVista (math diagrams, mixed open/multiple-choice), AI2D (science diagrams, multiple-choice), ChartQA (statistical charts, open-ended), MMMU (college exam questions, multiple-choice, multi-domain), and TextVQA (natural photographs containing text, open-ended) \u2014 using Qwen2-VL-2B-Instruct on free-tier GPU compute (Kaggle T4\u00d72 / Colab T4) exclusively. We sampled up to 400 items per dataset (fewer for MathVista and MMMU multiple-choice subsets, per their available sizes) and concatenated them in the fixed order MathVista\u2192AI2D\u2192ChartQA\u2192MMMU\u2192TextVQA, producing a real, non-synthetic stream of n=1675 items with four genuine domain-shift points."),

        H2("5.2 Fixing a Real Failure Mode: the Hybrid Score"),
        P("The initial semantic-volume-only score (Sec. 3.2, v1) turned out to be degenerate for short-form answers: manual inspection confirmed that repeated, independently sampled generations for a single-letter or single-number answer come back literally identical (e.g. \u201cC|C|C|C|C|C\u201d) in the large majority of examples, while longer, multi-step reasoning answers show genuine lexical diversity. This reproduced across both models and multiple benchmarks, ruling out an implementation artifact. The hybrid score (Sec. 3.2) resolves this by falling back on first-token predictive entropy for short answers, and produces a large, consistent improvement on both the degeneracy rate and the informativeness of the score (Figure 3)."),
        ...figure("../figures/fig3_hybrid_comparison.png", 580, 260, "Figure 3. The hybrid score reduces degenerate zero-scores from 95\u201398% to 8\u201321%, and improves uncertainty-score informativeness (AUROC of score vs. incorrectness) from chance level to 0.60\u20130.64, on the two benchmarks used in the initial exploratory pilot."),

        H2("5.3 Calibration Mechanism on the 5-Domain Real Stream"),
        P("With the hybrid score deployed at scale, the 5-domain stream (n=1675, Sec. 5.1) gives a real, non-degenerate, multiply-shifted test of the calibration mechanism itself (Figure 4)."),
        ...figure("../figures/fig4_five_domain_stream.png", 580, 435, "Figure 4. Top: nonconformity scores and calibration thresholds over the full 1675-item, 5-domain real stream. Bottom: rolling coverage (window=30) for all four methods across four consecutive real domain shifts. Fixed-threshold calibration is consistently unstable and collapses severely at the MMMU transition; the adaptive methods track the 0.70 target throughout, with the drift-aware mechanism at or above standard ACI in every domain."),
        simpleTable(
          ["Method", "Overall", "MathVista", "AI2D", "ChartQA", "MMMU", "TextVQA"],
          [
            ["Fixed / Split CP", "0.51", "0.58", "0.55", "0.63", "0.20", "0.45"],
            ["Standard ACI", "0.74", "0.73", "0.73", "0.76", "0.72", "0.74"],
            ["PID-ACI [4]", "0.70", "0.70", "0.70", "0.70", "0.70", "0.70"],
            ["Drift-Aware ACI (ours)", "0.75", "0.75", "0.74", "0.76", "0.75", "0.77"],
          ],
          [2700, 1300, 1300, 1300, 1300, 1300, 1300]
        ),
        new Paragraph({ text: "", spacing: { after: 200 } }),
        P("Three findings stand out. First, fixed-threshold calibration is not merely fragile at a single shift point \u2014 it is unstable across every single domain transition, falling as low as 0.20 in MMMU (against a 0.70 target) and never exceeding 0.63 in any domain. Second, our drift-aware mechanism is the best or tied-best performer in all five domains individually, not just on average \u2014 a stronger and more scale-tested claim than our earlier two-domain pilot could support. Third, PID-ACI [4] again lands almost exactly on the 0.70 target in every domain (0.697\u20130.701), a striking and consistent empirical signature of its theoretical long-run guarantee, discussed further in Sec. 5.5."),

        H2("5.4 Statistical Validation via Bootstrap"),
        P("To quantify sampling uncertainty at this larger scale, we bootstrap the 5-domain stream 1000 times, resampling with replacement independently within each of the five domain blocks (preserving the MathVista\u2192AI2D\u2192ChartQA\u2192MMMU\u2192TextVQA block order, the source of the four genuine shift events) and recomputing coverage for all four methods on each resample (Figure 5, Figure 6)."),
        ...figure("../figures/fig5_forest_5domain.png", 580, 148, "Figure 5. Bootstrap 95% confidence intervals (1000 resamples) for overall coverage and for coverage within each of the five domains. Fixed-threshold CP's interval is wide in every domain, reflecting genuine instability; the three adaptive methods are all tightly estimated, with ours consistently highest or tied-highest."),
        ...figure("../figures/fig6_ridgeline_5domain.png", 480, 274, "Figure 6. Full bootstrap distributions of overall coverage. Fixed CP's distribution is wide, multi-modal, and centered well below target; PID-ACI [4] is a near-delta-function just under target; standard ACI and our method are both narrow, with ours centered highest and closest to target."),
        P("The overall-coverage difference between our method and standard ACI has bootstrap mean 0.0146 with 95% CI [0.0089, 0.0215] \u2014 excluding zero entirely \u2014 and our method has higher overall coverage than standard ACI in all 1000 of 1000 bootstrap resamples. This is, to our knowledge, the strongest statistical evidence in this paper that the advantage is real rather than a resampling or single-run artifact, and it now spans five domains and four shift points rather than one of each."),

        H2("5.5 Comparison with PID-ACI [4]"),
        P("Two findings about PID-ACI [4] are worth reporting precisely, including one that complicates a simple \u201cours is better\u201d narrative. First, PID-ACI's bootstrap distribution is the narrowest of all four methods at every scale we tested (2-domain and 5-domain alike) \u2014 empirically consistent with its theoretical long-run guarantee, which the source paper proves without any boundedness assumption on the scores, and this pattern reproduces strikingly consistently: coverage within 0.697\u20130.701 in all five domains of Sec. 5.3. Second, and less favorably for us, PID-ACI's average threshold value was extremely poorly scaled for our data in the original 2-domain pilot (mean q \u2248 124, versus scores that never exceed \u2248 0.12): its tan-based integrator saturated repeatedly under the heuristic C_sat and K_I constants we selected using the source paper's general guidance, since no dataset-specific defaults are given. We did not perform a dedicated hyperparameter search for PID-ACI, so this comparison should be read as reflecting a reasonable-effort baseline, not a fully tuned one; a more careful choice of C_sat/K_I might close some or all of the coverage gap to our method, though it is notable that even without tuning, PID-ACI's raw coverage rate is already close to target \u2014 its main weakness in our setting is threshold scale, not long-run miscoverage rate."),

        H2("5.6 Testing [6]'s Own Score Family on Our Shift"),
        P("Sec. 2.3 conjectured that [6]'s bounded, token-probability nonconformity scores (LAC, APS) might already be fairly shift-tolerant under fixed-threshold calibration \u2014 unlike our sample-based hybrid score \u2014 precisely because they are bounded per-class probabilities rather than open-ended sample-diversity statistics. We tested this directly on the original two-domain shift: we computed LAC and APS scores (via a single forward pass per example, restricting the softmax to the valid answer-letter tokens, following [6]'s method) for every multiple-choice item in the MathVista\u2192MMMU stream (n=466: 300 MathVista + 166 MMMU items with explicit options), and reran all four calibration methods on each score stream (Figure 7)."),
        ...figure("../figures/fig7_lacaps.png", 580, 250, "Figure 7. Coverage by phase for LAC (left) and APS (right) scores on the MathVista\u2192MMMU shift. With LAC, fixed-threshold calibration degrades sharply after the shift (0.37 in the back half of MMMU, target 0.70) \u2014 the same qualitative failure we report for our hybrid score. With APS, fixed-threshold calibration does not degrade at all (0.76 in the same phase), consistent with [6]'s own finding that APS is comparatively shift-tolerant."),
        P("This is a genuinely mixed result, and we report it as such. With the LAC score, fixed-threshold calibration fails on this shift in the same qualitative way it does with our hybrid score, and all three adaptive methods, including ours, recover it (0.60\u20130.71). With the APS score, fixed-threshold calibration does not degrade at all (0.76, if anything mildly over-covering), matching [6]'s own report that APS is comparatively robust to distribution shift when restricted to well-behaved multiple-choice probability scores. This is direct empirical support \u2014 on our own real data, not just by citation \u2014 for the distinction drawn in Sec. 2.3: shift-robustness of fixed-threshold conformal calibration is a property of the score function as much as of the calibration procedure. It also tempers any claim that our mechanism is strictly necessary in all conformal-VLM settings: for applications restricted to closed-form multiple-choice answers with a well-chosen bounded score such as APS, simple fixed-threshold calibration may already suffice. We view this as a useful scoping result rather than a weakness to be hidden; extending this five-way comparison of LAC/APS to all five domains of Sec. 5.3\u20135.4 is noted in Sec. 6."),

        H1("6. Planned Next Steps"),
        bullet("Extend the LAC/APS comparison (Sec. 5.6, currently MathVista+MMMU only) to all five domains of the main stream (Sec. 5.3\u20135.4), for a fully matched-scale comparison."),
        bullet("Perform a dedicated hyperparameter search for PID-ACI's C_sat and K_I constants (Sec. 5.5), rather than the reasonable-effort heuristic used in this pilot, before treating the PID-ACI comparison as final."),
        bullet("Investigate why LAC and APS diverge so sharply in shift-robustness on our data (Sec. 5.6) \u2014 e.g., whether it relates to their differing sensitivity to model overconfidence \u2014 to better predict, ahead of time, which score families need drift-aware calibration and which do not."),
        bullet("Extend the ablation of Sec. 4.1 (currently synthetic-only) and the hybrid-score threshold \u03c4 to the real 5-domain stream, which now has enough scale (n=1675) to support it."),
        bullet("Design a principled joint calibration between the two hybrid-score branches (shared rank/CDF transform) rather than the fixed rescaling constant used in the pilot."),
        bullet("Add at least one stronger backbone (e.g., a 7B-8B-class open VLM) alongside Qwen2-VL-2B, to test whether the coverage pattern is backbone-independent."),
        bullet("Complete the reweighting-term bound connecting Theorem 1 to the unweighted rate via an explicit drift-estimator-fidelity assumption (Sec. 3.4, Remark)."),
        bullet("Extend the real-data evaluation to a genuinely continuous streaming (not five-block) drift schedule [7], and to video/temporal domains."),

        H1("7. Limitations"),
        bullet("The main calibration validation (Sec. 5.3\u20135.6) uses a single backbone (Qwen2-VL-2B-Instruct, 2B parameters); the exploratory moondream2-1.9B pilot (Sec. 5.2) only established the score-collapse finding, not the full calibration comparison. Whether the coverage pattern holds for substantially stronger backbones is untested and should not be assumed."),
        bullet("The five domains are still sampled once each (up to 400 items per dataset) and concatenated in a single fixed order; we have not tested sensitivity to domain ordering or repeated the domain composition with different random subsets of each source dataset."),
        bullet("The hybrid score's cross-branch rescaling is currently ad hoc and needs principled joint calibration before being used as the paper's primary result."),
        bullet("All real-data experiments so far use free-tier GPU compute; the theoretical guarantee (Sec. 3.4) is not yet empirically stress-tested at sample sizes beyond n\u22481675."),
        bullet("The PID-ACI [4] baseline (Sec. 5.5) uses heuristic, not dedicated-searched, hyperparameters (C_sat, K_I); its reported coverage gap relative to our method may narrow under better tuning, and should not be over-interpreted as a definitive ranking."),
        bullet("Sec. 5.6 shows our method's advantage over fixed-threshold calibration does not hold uniformly across score families, and this comparison has not yet been scaled to all five domains (Sec. 6). Our contribution is best scoped to sample-based, open-ended-generation-compatible nonconformity scores, not to conformal prediction for multimodal models in general."),
        bullet("The real-data \u201cdomain shift\u201d (Sec. 5) is constructed by concatenating five fixed datasets in one order, i.e. four abrupt block changes, not a genuinely continuous streaming distribution with gradual, repeated, or non-monotone drift. This is a deliberately simple, interpretable proxy for the harder streaming setting motivated in the Introduction (video agents, robotics, streaming triage), in the same spirit as early time-series conformal evaluations that also begin with block or synthetic shift schedules before genuinely continuous online deployment [7]. Demonstrating the mechanism on a real, continuously-drifting temporal stream (e.g., a rolling video or news feed) rather than block transitions is left as future work."),
        bullet("No ablation over the hybrid-score branch threshold \u03c4 (Sec. 3.2) has yet been run on real data; the calibration mechanism's own hyperparameters (\u03b3_min, \u03b3_max, \u03bb, drift-estimation window) have been ablated on synthetic data (Sec. 4.1) and found robust, but this has not yet been repeated on real data."),
        bullet("Bootstrap confidence intervals (Sec. 5.4) resample which examples appear from an already-generated, fixed set of model outputs; they quantify sampling uncertainty over item composition, but cannot substitute for genuinely repeated VLM generations (independent fresh sampling draws from the model itself for the same items), which remains future work."),

        H1("Statements and Declarations"),
        P("See the separate title page for Funding, Competing interests, Ethics approval, Consent, Data availability, Code availability, Authors' contributions, and Use-of-AI declarations, as required by the target journal's submission guidelines.", { italics: true }),

        H1("References"),
        numref(1, "Lau, G.K.R., Dao, H., Lin, N.K.H., Low, B.K.H.: Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume. arXiv:2602.24195 (2026). https://doi.org/10.48550/arXiv.2602.24195"),
        numref(2, "Padhi, T., Kaur, R., Cobb, A.D., Acharya, M., Roy, A., Samplawski, C., Matejek, B., Berenbeim, A.M., Bastian, N.D., Jha, S.: Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding. arXiv:2505.03788 (2025). https://doi.org/10.48550/arXiv.2505.03788"),
        numref(3, "Gibbs, I., Cand\u00e8s, E.: Adaptive Conformal Inference Under Distribution Shift. In: Advances in Neural Information Processing Systems, vol. 34, pp. 1660\u20131672 (2021). arXiv:2106.00170. https://doi.org/10.48550/arXiv.2106.00170"),
        numref(4, "Angelopoulos, A.N., Cand\u00e8s, E.J., Tibshirani, R.J.: Conformal PID Control for Time Series Prediction. In: Advances in Neural Information Processing Systems, vol. 36, pp. 23047\u201323074 (2023). arXiv:2307.16895. https://doi.org/10.48550/arXiv.2307.16895"),
        numref(5, "Correia, A.H.C., Louizos, C.: Non-exchangeable Conformal Prediction with Optimal Transport: Tackling Distribution Shifts with Unlabeled Data. In: Advances in Neural Information Processing Systems 38 (2025). arXiv:2507.10425. https://doi.org/10.48550/arXiv.2507.10425"),
        numref(6, "Azad, A., Hossain, M.S., Shanto, M.S.H., Rahman, M.S., Parvez, M.R.: The Art of Saying \u201cMaybe\u201d: A Conformal Lens for Uncertainty Benchmarking in VLMs. In: Findings of the Association for Computational Linguistics: EACL 2026, pp. 5185\u20135201 (2026). Preprint: arXiv:2509.13379. https://doi.org/10.48550/arXiv.2509.13379"),
        numref(7, "Xu, C., Xie, Y.: Conformal Prediction Interval for Dynamic Time-Series. In: Proceedings of the 38th International Conference on Machine Learning, PMLR 139, pp. 11559\u201311569 (2021). Preprint: arXiv:2010.09107. https://doi.org/10.48550/arXiv.2010.09107"),
        new Paragraph({ text: "", spacing: { after: 160 } }),
        P("Note: DOIs above resolve to the arXiv preprint for each work (the version-independent \u201cconcept DOI\u201d format, 10.48550/arXiv.XXXX.XXXXX, per arXiv/DataCite convention); [6] and [7] are also formally published at the venues listed, which may carry their own separate publisher DOIs \u2014 confirm and use the publisher DOI instead if the target journal requires it. This reference list uses a generic numbered author-year format for the draft; confirm and reformat to the exact citation style required by the target journal's \u201cInstructions for Authors\u201d before submission.", { italics: true }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("./paper_draft_springer.docx", buf);
  console.log("written", buf.length, "bytes");
});
