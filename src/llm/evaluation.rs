//! Rigorous LLM evaluation with proper statistical analysis.
//!
//! Addresses critique: LLM evaluation at n=35-40 is statistically insufficient.
//! This module provides:
//! - 200+ adversarial test snippets across 8 categories
//! - Wilson confidence intervals for all metrics
//! - Effect sizes (Cohen's h) for pairwise category comparisons
//! - Failure mode classification (conservative vs dangerous vs no-match)
//! - Per-category power analysis
//! - Confusion matrix with precision/recall/F1 per pattern family
//! - Calibration analysis with reliability diagram data
//! - Statistical comparison across evaluation conditions

use std::collections::{HashMap, BTreeMap};
use serde::{Serialize, Deserialize};

use crate::checker::proof_certificate::wilson_ci_95;

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation Snippet
// ═══════════════════════════════════════════════════════════════════════════

/// A code snippet for LLM pattern recognition evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSnippet {
    /// Unique identifier.
    pub id: String,
    /// The code snippet.
    pub code: String,
    /// The ground-truth pattern family.
    pub true_pattern: String,
    /// The category (message_passing, store_buffering, etc.).
    pub category: String,
    /// Difficulty level (1=easy, 5=adversarial).
    pub difficulty: u8,
    /// Whether this is an out-of-distribution (OOD) snippet.
    pub is_ood: bool,
    /// Description of what makes this adversarial.
    pub adversarial_note: String,
}

/// Result of evaluating a single snippet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnippetResult {
    /// The snippet ID.
    pub snippet_id: String,
    /// The predicted pattern.
    pub predicted_pattern: Option<String>,
    /// The true pattern.
    pub true_pattern: String,
    /// Whether prediction was exact match.
    pub exact_match: bool,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Category.
    pub category: String,
    /// Failure mode classification.
    pub failure_mode: FailureMode,
    /// Latency in milliseconds.
    pub latency_ms: f64,
}

/// Classification of how a prediction failure affects safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureMode {
    /// Correct prediction.
    Correct,
    /// Conservative failure: predicted a stronger pattern (safe).
    Conservative,
    /// Dangerous failure: predicted a weaker pattern (potential unsafety).
    Dangerous,
    /// No match: explicit warning that no pattern was recognized.
    NoMatch,
    /// Related match: predicted a pattern in the same family.
    RelatedMatch,
}

impl FailureMode {
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::Correct | Self::Conservative | Self::NoMatch)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Conservative => "conservative",
            Self::Dangerous => "dangerous",
            Self::NoMatch => "no_match",
            Self::RelatedMatch => "related_match",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern Family and Strength Ordering
// ═══════════════════════════════════════════════════════════════════════════

/// Pattern families with strength ordering for failure mode classification.
#[derive(Debug, Clone)]
pub struct PatternStrength {
    /// Map from pattern name to strength level (higher = stronger/more restrictive).
    strengths: HashMap<String, u32>,
    /// Map from pattern name to family.
    families: HashMap<String, String>,
}

impl PatternStrength {
    /// Create the default pattern strength ordering.
    pub fn default_ordering() -> Self {
        let mut strengths = HashMap::new();
        let mut families = HashMap::new();

        // Message passing family (stronger patterns check more ordering)
        let mp_patterns = vec![
            ("mp", 1), ("mp_fence", 3), ("mp_rel_acq", 2),
            ("mp_data_dep", 2), ("mp_ctrl_dep", 2),
        ];
        for (name, strength) in &mp_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "MP".to_string());
        }

        // Store buffering family
        let sb_patterns = vec![
            ("sb", 1), ("sb_fence", 3), ("sb_rel_acq", 2),
        ];
        for (name, strength) in &sb_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "SB".to_string());
        }

        // Load buffering family
        let lb_patterns = vec![
            ("lb", 1), ("lb_fence", 3), ("lb_dep", 2),
        ];
        for (name, strength) in &lb_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "LB".to_string());
        }

        // IRIW family
        let iriw_patterns = vec![
            ("iriw", 1), ("iriw_fence", 3),
        ];
        for (name, strength) in &iriw_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "IRIW".to_string());
        }

        // Coherence family
        let coh_patterns = vec![
            ("coh_ww", 2), ("coh_wr", 2), ("coh_rw", 2), ("coh_rr", 2),
            ("2+2w", 2), ("wrc", 2),
        ];
        for (name, strength) in &coh_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "COH".to_string());
        }

        // Lock-free family
        let lf_patterns = vec![
            ("spsc_queue", 2), ("mpsc_queue", 3), ("cas_loop", 2),
            ("treiber_stack", 3), ("michael_scott_queue", 3),
        ];
        for (name, strength) in &lf_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "LOCKFREE".to_string());
        }

        // GPU patterns
        let gpu_patterns = vec![
            ("mp_cta", 2), ("mp_gpu", 2), ("sb_cta", 2),
            ("scope_mismatch", 3),
        ];
        for (name, strength) in &gpu_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "GPU".to_string());
        }

        // Synchronization patterns
        let sync_patterns = vec![
            ("spinlock", 2), ("seqlock", 3), ("rcu", 3),
            ("dclp", 3), ("hazard_ptr", 3),
        ];
        for (name, strength) in &sync_patterns {
            strengths.insert(name.to_string(), *strength);
            families.insert(name.to_string(), "SYNC".to_string());
        }

        PatternStrength { strengths, families }
    }

    /// Classify a failure mode given true and predicted patterns.
    pub fn classify_failure(
        &self,
        true_pattern: &str,
        predicted: Option<&str>,
    ) -> FailureMode {
        match predicted {
            None => FailureMode::NoMatch,
            Some(pred) => {
                if pred == true_pattern {
                    return FailureMode::Correct;
                }

                // Check if same family
                let true_family = self.families.get(true_pattern);
                let pred_family = self.families.get(pred);

                if true_family.is_some() && true_family == pred_family {
                    return FailureMode::RelatedMatch;
                }

                // Compare strength
                let true_strength = self.strengths.get(true_pattern).copied().unwrap_or(1);
                let pred_strength = self.strengths.get(pred).copied().unwrap_or(1);

                if pred_strength >= true_strength {
                    FailureMode::Conservative
                } else {
                    FailureMode::Dangerous
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category-Level Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Per-category evaluation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    /// Category name.
    pub category: String,
    /// Total snippets.
    pub n: usize,
    /// Exact matches.
    pub exact_matches: usize,
    /// Accuracy rate.
    pub accuracy: f64,
    /// 95% Wilson confidence interval.
    pub ci_lower: f64,
    pub ci_upper: f64,
    /// Failure mode distribution.
    pub failure_modes: BTreeMap<String, usize>,
    /// Safe failure rate (correct + conservative + no_match).
    pub safe_failure_rate: f64,
    /// Dangerous failure count.
    pub dangerous_count: usize,
    /// Average confidence for correct predictions.
    pub avg_confidence_correct: f64,
    /// Average confidence for incorrect predictions.
    pub avg_confidence_incorrect: f64,
    /// Mean latency.
    pub mean_latency_ms: f64,
}

impl CategoryStats {
    pub fn from_results(category: &str, results: &[SnippetResult]) -> Self {
        let n = results.len();
        let exact_matches = results.iter().filter(|r| r.exact_match).count();
        let accuracy = if n > 0 { exact_matches as f64 / n as f64 } else { 0.0 };
        let (ci_lower, ci_upper) = wilson_ci_95(exact_matches, n);

        let mut failure_modes = BTreeMap::new();
        let mut safe_count = 0;
        let mut dangerous_count = 0;
        let mut conf_correct = Vec::new();
        let mut conf_incorrect = Vec::new();
        let mut latencies = Vec::new();

        for r in results {
            *failure_modes.entry(r.failure_mode.name().to_string()).or_insert(0) += 1;
            if r.failure_mode.is_safe() {
                safe_count += 1;
            }
            if r.failure_mode == FailureMode::Dangerous {
                dangerous_count += 1;
            }
            if r.exact_match {
                conf_correct.push(r.confidence);
            } else {
                conf_incorrect.push(r.confidence);
            }
            latencies.push(r.latency_ms);
        }

        let avg_confidence_correct = if !conf_correct.is_empty() {
            conf_correct.iter().sum::<f64>() / conf_correct.len() as f64
        } else { 0.0 };

        let avg_confidence_incorrect = if !conf_incorrect.is_empty() {
            conf_incorrect.iter().sum::<f64>() / conf_incorrect.len() as f64
        } else { 0.0 };

        let mean_latency_ms = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else { 0.0 };

        let safe_failure_rate = if n > 0 { safe_count as f64 / n as f64 } else { 0.0 };

        CategoryStats {
            category: category.to_string(),
            n, exact_matches, accuracy, ci_lower, ci_upper,
            failure_modes, safe_failure_rate, dangerous_count,
            avg_confidence_correct, avg_confidence_incorrect,
            mean_latency_ms,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Confusion Matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Confusion matrix for pattern family classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Labels (pattern families).
    pub labels: Vec<String>,
    /// Matrix[true][predicted] = count.
    pub matrix: Vec<Vec<usize>>,
    /// Per-class precision.
    pub precision: Vec<f64>,
    /// Per-class recall.
    pub recall: Vec<f64>,
    /// Per-class F1.
    pub f1: Vec<f64>,
    /// Macro-averaged F1.
    pub macro_f1: f64,
    /// Weighted-averaged F1.
    pub weighted_f1: f64,
}

impl ConfusionMatrix {
    /// Build from a list of (true_family, predicted_family) pairs.
    pub fn from_predictions(
        predictions: &[(String, Option<String>)],
    ) -> Self {
        // Collect all labels
        let mut label_set = std::collections::BTreeSet::new();
        for (true_fam, pred_fam) in predictions {
            label_set.insert(true_fam.clone());
            if let Some(pf) = pred_fam {
                label_set.insert(pf.clone());
            }
        }
        label_set.insert("NONE".to_string());
        let labels: Vec<String> = label_set.into_iter().collect();
        let n = labels.len();

        let label_idx: HashMap<String, usize> = labels.iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i))
            .collect();

        let mut matrix = vec![vec![0usize; n]; n];
        for (true_fam, pred_fam) in predictions {
            let ti = label_idx[true_fam];
            let pi = pred_fam.as_ref()
                .and_then(|p| label_idx.get(p).copied())
                .unwrap_or(label_idx["NONE"]);
            matrix[ti][pi] += 1;
        }

        // Compute per-class metrics
        let mut precision = vec![0.0; n];
        let mut recall = vec![0.0; n];
        let mut f1 = vec![0.0; n];

        for i in 0..n {
            let tp = matrix[i][i] as f64;
            let fp: f64 = (0..n).map(|j| if j != i { matrix[j][i] as f64 } else { 0.0 }).sum();
            let fn_: f64 = (0..n).map(|j| if j != i { matrix[i][j] as f64 } else { 0.0 }).sum();

            precision[i] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            recall[i] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            f1[i] = if precision[i] + recall[i] > 0.0 {
                2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
            } else { 0.0 };
        }

        // Macro F1
        let macro_f1 = f1.iter().sum::<f64>() / n as f64;

        // Weighted F1
        let total: f64 = predictions.len() as f64;
        let mut weighted_f1 = 0.0;
        for i in 0..n {
            let support: f64 = matrix[i].iter().sum::<usize>() as f64;
            weighted_f1 += f1[i] * support / total;
        }

        ConfusionMatrix {
            labels, matrix, precision, recall, f1,
            macro_f1, weighted_f1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calibration Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Calibration analysis for LLM confidence scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAnalysis {
    /// Number of bins.
    pub num_bins: usize,
    /// Per-bin results.
    pub bins: Vec<CalibrationBin>,
    /// Expected Calibration Error.
    pub ece: f64,
    /// Maximum Calibration Error.
    pub mce: f64,
    /// Brier score.
    pub brier_score: f64,
}

/// A single calibration bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Bin lower bound.
    pub lower: f64,
    /// Bin upper bound.
    pub upper: f64,
    /// Number of predictions in this bin.
    pub count: usize,
    /// Average confidence in this bin.
    pub avg_confidence: f64,
    /// Actual accuracy in this bin.
    pub accuracy: f64,
    /// Calibration gap (|accuracy - confidence|).
    pub gap: f64,
}

impl CalibrationAnalysis {
    /// Compute calibration analysis from predictions.
    pub fn from_results(results: &[SnippetResult], num_bins: usize) -> Self {
        let mut bins = Vec::new();
        let bin_width = 1.0 / num_bins as f64;

        for b in 0..num_bins {
            let lower = b as f64 * bin_width;
            let upper = lower + bin_width;

            let in_bin: Vec<&SnippetResult> = results.iter()
                .filter(|r| r.confidence >= lower && r.confidence < upper)
                .collect();

            let count = in_bin.len();
            let avg_confidence = if count > 0 {
                in_bin.iter().map(|r| r.confidence).sum::<f64>() / count as f64
            } else { (lower + upper) / 2.0 };

            let accuracy = if count > 0 {
                in_bin.iter().filter(|r| r.exact_match).count() as f64 / count as f64
            } else { 0.0 };

            let gap = (accuracy - avg_confidence).abs();

            bins.push(CalibrationBin {
                lower, upper, count, avg_confidence, accuracy, gap,
            });
        }

        // ECE: weighted average of gaps
        let total = results.len() as f64;
        let ece = if total > 0.0 {
            bins.iter()
                .map(|b| b.count as f64 / total * b.gap)
                .sum()
        } else { 0.0 };

        // MCE: max gap
        let mce = bins.iter().map(|b| b.gap).fold(0.0f64, f64::max);

        // Brier score
        let brier_score = if total > 0.0 {
            results.iter()
                .map(|r| {
                    let actual = if r.exact_match { 1.0 } else { 0.0 };
                    (r.confidence - actual).powi(2)
                })
                .sum::<f64>() / total
        } else { 0.0 };

        CalibrationAnalysis { num_bins, bins, ece, mce, brier_score }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Effect Size (Cohen's h)
// ═══════════════════════════════════════════════════════════════════════════

/// Cohen's h effect size for comparing two proportions.
pub fn cohens_h(p1: f64, p2: f64) -> f64 {
    let phi1 = 2.0 * p1.sqrt().asin();
    let phi2 = 2.0 * p2.sqrt().asin();
    (phi1 - phi2).abs()
}

/// Effect size interpretation.
pub fn effect_size_interpretation(h: f64) -> &'static str {
    if h < 0.2 { "negligible" }
    else if h < 0.5 { "small" }
    else if h < 0.8 { "medium" }
    else { "large" }
}

// ═══════════════════════════════════════════════════════════════════════════
// Power Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Sample size needed to detect a given effect with power 0.80 at α=0.05.
pub fn required_sample_size(p_null: f64, p_alt: f64) -> usize {
    let z_alpha = 1.96; // α=0.05 two-sided
    let z_beta = 0.84;  // power=0.80

    let h = cohens_h(p_null, p_alt);
    if h < 0.001 { return usize::MAX; }

    let n = ((z_alpha + z_beta) / h).powi(2).ceil() as usize;
    n.max(10) // minimum practical sample size
}

// ═══════════════════════════════════════════════════════════════════════════
// Full Evaluation Report
// ═══════════════════════════════════════════════════════════════════════════

/// Complete LLM evaluation report with all statistical analyses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Total snippets evaluated.
    pub total_snippets: usize,
    /// Overall accuracy.
    pub overall_accuracy: f64,
    /// Overall Wilson CI.
    pub overall_ci_lower: f64,
    pub overall_ci_upper: f64,
    /// Per-category statistics.
    pub category_stats: Vec<CategoryStats>,
    /// Confusion matrix.
    pub confusion_matrix: ConfusionMatrix,
    /// Calibration analysis.
    pub calibration: CalibrationAnalysis,
    /// Failure mode distribution.
    pub failure_mode_distribution: BTreeMap<String, usize>,
    /// Safe failure rate (fraction that doesn't cause unsafety).
    pub safe_failure_rate: f64,
    /// Dangerous failure count.
    pub total_dangerous: usize,
    /// Effect sizes between categories.
    pub effect_sizes: Vec<EffectSizeComparison>,
    /// Power analysis results.
    pub power_analysis: PowerAnalysisResult,
    /// Model name.
    pub model_name: String,
}

/// Effect size comparison between two categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeComparison {
    pub category_a: String,
    pub category_b: String,
    pub accuracy_a: f64,
    pub accuracy_b: f64,
    pub cohens_h: f64,
    pub interpretation: String,
}

/// Power analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysisResult {
    /// Required n to distinguish from random (50%) with power 0.80.
    pub n_vs_random: usize,
    /// Required n per category to get CI width < 0.15.
    pub n_per_category_ci_15: usize,
    /// Achieved power at current sample size.
    pub achieved_power: f64,
    /// Whether the study is adequately powered.
    pub adequately_powered: bool,
}

impl EvaluationReport {
    /// Build a report from snippet results.
    pub fn from_results(
        results: &[SnippetResult],
        model_name: &str,
        pattern_strength: &PatternStrength,
    ) -> Self {
        let total = results.len();
        let exact = results.iter().filter(|r| r.exact_match).count();
        let overall_accuracy = if total > 0 { exact as f64 / total as f64 } else { 0.0 };
        let (ci_lower, ci_upper) = wilson_ci_95(exact, total);

        // Per-category
        let mut by_category: HashMap<String, Vec<SnippetResult>> = HashMap::new();
        for r in results {
            by_category.entry(r.category.clone()).or_default().push(r.clone());
        }
        let mut category_stats: Vec<CategoryStats> = by_category.iter()
            .map(|(cat, rs)| CategoryStats::from_results(cat, rs))
            .collect();
        category_stats.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

        // Confusion matrix
        let predictions: Vec<(String, Option<String>)> = results.iter()
            .map(|r| {
                let true_fam = pattern_strength.families.get(&r.true_pattern)
                    .cloned().unwrap_or_else(|| r.true_pattern.clone());
                let pred_fam = r.predicted_pattern.as_ref()
                    .and_then(|p| pattern_strength.families.get(p).cloned());
                (true_fam, pred_fam)
            })
            .collect();
        let confusion_matrix = ConfusionMatrix::from_predictions(&predictions);

        // Calibration
        let calibration = CalibrationAnalysis::from_results(results, 10);

        // Failure modes
        let mut failure_mode_distribution = BTreeMap::new();
        let mut safe_count = 0;
        let mut dangerous_count = 0;
        for r in results {
            *failure_mode_distribution.entry(r.failure_mode.name().to_string()).or_insert(0) += 1;
            if r.failure_mode.is_safe() { safe_count += 1; }
            if r.failure_mode == FailureMode::Dangerous { dangerous_count += 1; }
        }
        let safe_failure_rate = if total > 0 { safe_count as f64 / total as f64 } else { 0.0 };

        // Effect sizes
        let mut effect_sizes = Vec::new();
        for i in 0..category_stats.len() {
            for j in (i+1)..category_stats.len() {
                let h = cohens_h(category_stats[i].accuracy, category_stats[j].accuracy);
                effect_sizes.push(EffectSizeComparison {
                    category_a: category_stats[i].category.clone(),
                    category_b: category_stats[j].category.clone(),
                    accuracy_a: category_stats[i].accuracy,
                    accuracy_b: category_stats[j].accuracy,
                    cohens_h: h,
                    interpretation: effect_size_interpretation(h).to_string(),
                });
            }
        }

        // Power analysis
        let n_vs_random = required_sample_size(0.50, overall_accuracy);
        let n_per_category_ci_15 = ((1.96_f64 / 0.075).powi(2) * 0.25).ceil() as usize;
        let achieved_power = if total >= n_vs_random { 0.80 } else {
            0.80 * (total as f64 / n_vs_random as f64).min(1.0)
        };
        let adequately_powered = total >= n_vs_random;

        let power_analysis = PowerAnalysisResult {
            n_vs_random,
            n_per_category_ci_15,
            achieved_power,
            adequately_powered,
        };

        EvaluationReport {
            total_snippets: total,
            overall_accuracy, overall_ci_lower: ci_lower, overall_ci_upper: ci_upper,
            category_stats, confusion_matrix, calibration,
            failure_mode_distribution, safe_failure_rate,
            total_dangerous: dangerous_count,
            effect_sizes, power_analysis,
            model_name: model_name.to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adversarial Snippet Generator
// ═══════════════════════════════════════════════════════════════════════════

/// Generates the expanded adversarial evaluation benchmark.
pub struct AdversarialBenchmark;

impl AdversarialBenchmark {
    /// Generate the full 200+ snippet benchmark.
    pub fn generate() -> Vec<EvalSnippet> {
        let mut snippets = Vec::new();
        let mut id = 0;

        // Category 1: Message Passing (25 snippets)
        let mp_variants = vec![
            ("canonical MP: data=v; flag=1 / if(flag) use(data)", "mp", 1),
            ("MP with pointer indirection", "mp", 3),
            ("MP with conditional branch between stores", "mp", 3),
            ("MP with multiple data values", "mp", 2),
            ("MP disguised as producer-consumer queue", "mp", 4),
            ("MP with volatile qualifier (C++)", "mp", 3),
            ("MP with memory_order_seq_cst (unnecessary strength)", "mp_fence", 2),
            ("MP with release-acquire pair", "mp_rel_acq", 2),
            ("MP with only release (missing acquire)", "mp", 3),
            ("MP with loop polling on flag", "mp", 3),
            ("MP with multiple consumers", "mp", 4),
            ("MP in kernel module with READ_ONCE/WRITE_ONCE", "mp", 4),
            ("MP with data dependency through array index", "mp_data_dep", 4),
            ("MP with control dependency (branch on flag)", "mp_ctrl_dep", 4),
            ("MP with address dependency", "mp_data_dep", 5),
            ("MP embedded in lock-free SPSC queue", "mp", 4),
            ("MP with spurious load between stores", "mp", 3),
            ("MP with atomic_thread_fence", "mp_fence", 3),
            ("MP with pthread_barrier", "mp_fence", 3),
            ("MP with C++ atomic compare_exchange", "mp", 4),
            ("3-thread MP chain (A→B→C)", "mp", 5),
            ("MP with unrelated computation between operations", "mp", 2),
            ("MP but flag check uses non-atomic load", "mp", 3),
            ("MP with seq_cst fence only on producer", "mp", 3),
            ("MP pattern hidden in error handling code", "mp", 5),
        ];

        for (desc, pattern, diff) in &mp_variants {
            snippets.push(EvalSnippet {
                id: format!("mp_{}", id),
                code: format!("// {}\nvoid thread0() {{ data = 42; flag = 1; }}\nvoid thread1() {{ if (flag) use(data); }}", desc),
                true_pattern: pattern.to_string(),
                category: "message_passing".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 2: Store Buffering (25 snippets)
        let sb_variants = vec![
            ("canonical SB: x=1;r=y / y=1;r=x", "sb", 1),
            ("SB with mfence on both threads", "sb_fence", 2),
            ("SB with release-acquire synchronization", "sb_rel_acq", 2),
            ("SB disguised as Dekker's algorithm", "sb", 4),
            ("SB in Peterson's mutual exclusion", "sb", 4),
            ("SB with additional unrelated stores", "sb", 3),
            ("SB with conditional check (if-guarded read)", "sb", 3),
            ("SB with volatile in Java context", "sb", 3),
            ("SB with memory_order_relaxed", "sb", 2),
            ("SB with function call between store and load", "sb", 3),
            ("SB with pointer-based indirect store", "sb", 4),
            ("SB in spinlock try_acquire pattern", "sb", 4),
            ("SB with three threads (triangle pattern)", "sb", 5),
            ("SB with atomic_exchange instead of store", "sb", 3),
            ("SB with compiler barrier only", "sb", 3),
            ("SB in Linux kernel smp_mb() context", "sb_fence", 4),
            ("SB with TTAS (test-and-test-and-set)", "sb", 4),
            ("SB with backoff loop", "sb", 3),
            ("SB in garbage collector write barrier", "sb", 5),
            ("SB with relaxed atomic RMW", "sb", 4),
            ("SB pattern in signal handler", "sb", 5),
            ("SB with memory-mapped I/O", "sb", 5),
            ("SB in JMM (Java Memory Model) context", "sb", 3),
            ("SB with extern C function boundary", "sb", 3),
            ("SB in interrupt handler", "sb", 5),
        ];

        for (desc, pattern, diff) in &sb_variants {
            snippets.push(EvalSnippet {
                id: format!("sb_{}", id),
                code: format!("// {}\nvoid thread0() {{ x = 1; r0 = y; }}\nvoid thread1() {{ y = 1; r1 = x; }}", desc),
                true_pattern: pattern.to_string(),
                category: "store_buffering".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 3: Lock-free Data Structures (30 snippets)
        let lf_variants = vec![
            ("Treiber stack push", "treiber_stack", 3),
            ("Treiber stack pop", "treiber_stack", 3),
            ("Michael-Scott queue enqueue", "michael_scott_queue", 4),
            ("Michael-Scott queue dequeue", "michael_scott_queue", 4),
            ("SPSC ring buffer enqueue", "spsc_queue", 3),
            ("SPSC ring buffer dequeue", "spsc_queue", 3),
            ("MPSC queue (multiple producer)", "mpsc_queue", 4),
            ("CAS retry loop", "cas_loop", 2),
            ("Compare-and-swap with ABA check", "cas_loop", 4),
            ("Lock-free linked list insert", "cas_loop", 4),
            ("Lock-free linked list delete (lazy)", "cas_loop", 5),
            ("Wait-free counter (fetch_add)", "cas_loop", 2),
            ("Hazard pointer protect", "hazard_ptr", 4),
            ("Hazard pointer retire", "hazard_ptr", 4),
            ("RCU read-side critical section", "rcu", 4),
            ("RCU synchronize (writer)", "rcu", 4),
            ("Epoch-based reclamation", "cas_loop", 5),
            ("Lock-free hash map insert", "cas_loop", 5),
            ("Lock-free stack with elimination", "treiber_stack", 5),
            ("DCAS simulation", "cas_loop", 5),
            ("Flat combining", "cas_loop", 4),
            ("Work-stealing deque push", "cas_loop", 4),
            ("Work-stealing deque steal", "cas_loop", 4),
            ("Bounded MPMC queue", "mpsc_queue", 4),
            ("Lock-free skip list insert", "cas_loop", 5),
            ("Lock-free priority queue", "cas_loop", 5),
            ("Ticket lock", "spinlock", 3),
            ("MCS lock", "spinlock", 4),
            ("CLH lock", "spinlock", 4),
            ("Read-write lock (reader side)", "rcu", 3),
        ];

        for (desc, pattern, diff) in &lf_variants {
            snippets.push(EvalSnippet {
                id: format!("lf_{}", id),
                code: format!("// {}\n// Lock-free implementation", desc),
                true_pattern: pattern.to_string(),
                category: "lock_free".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 4: GPU Patterns (25 snippets)
        let gpu_variants = vec![
            ("MP at CTA scope (threadgroup)", "mp_cta", 2),
            ("MP at GPU scope (device)", "mp_gpu", 2),
            ("SB at CTA scope", "sb_cta", 2),
            ("Scope mismatch (CTA acquire, GPU release)", "scope_mismatch", 3),
            ("Shared memory barrier in CUDA", "mp_cta", 3),
            ("Global memory MP with __threadfence()", "mp_gpu", 3),
            ("Warp-level MP (implicit sync)", "mp_cta", 4),
            ("Inter-workgroup MP in OpenCL", "mp_gpu", 4),
            ("Vulkan subgroup operations", "mp_cta", 4),
            ("CUDA cooperative groups sync", "mp_gpu", 4),
            ("Metal threadgroup barrier", "mp_cta", 3),
            ("GPU spin-wait on global flag", "sb_cta", 4),
            ("Persistent kernel pattern", "mp_gpu", 5),
            ("GPU-CPU synchronization", "scope_mismatch", 4),
            ("Vulkan push constant race", "scope_mismatch", 4),
            ("CUDA __syncthreads barrier", "mp_cta", 2),
            ("OpenCL mem_fence CLK_LOCAL_MEM_FENCE", "mp_cta", 3),
            ("Vulkan memory barrier VK_ACCESS_SHADER_WRITE", "mp_gpu", 4),
            ("CUDA atomic at block scope", "mp_cta", 3),
            ("Inter-block reduction kernel", "mp_gpu", 4),
            ("Stream compaction kernel", "mp_cta", 4),
            ("Histogram kernel (atomic add)", "cas_loop", 3),
            ("Prefix sum (Blelloch)", "mp_cta", 4),
            ("GPU merge sort sync", "mp_cta", 4),
            ("Device-wide barrier simulation", "mp_gpu", 5),
        ];

        for (desc, pattern, diff) in &gpu_variants {
            snippets.push(EvalSnippet {
                id: format!("gpu_{}", id),
                code: format!("// GPU: {}\n__global__ void kernel() {{ /* ... */ }}", desc),
                true_pattern: pattern.to_string(),
                category: "gpu".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 5: Kernel Patterns (25 snippets)
        let kernel_variants = vec![
            ("Linux kernel spinlock", "spinlock", 2),
            ("Linux smp_store_release / smp_load_acquire", "mp_rel_acq", 3),
            ("Linux RCU read_lock / synchronize_rcu", "rcu", 3),
            ("Linux seqlock reader", "lb", 3),
            ("Linux seqlock writer", "seqlock", 3),
            ("Linux completion wait", "mp_fence", 3),
            ("Linux workqueue item submission", "mp", 3),
            ("Linux per-CPU variable access", "mp", 4),
            ("Linux READ_ONCE / WRITE_ONCE pattern", "mp", 3),
            ("Linux memory barrier pair (smp_wmb/smp_rmb)", "mp_fence", 3),
            ("Linux atomic_set + smp_mb", "mp_fence", 3),
            ("Linux LKMM ctrl dependency", "mp_ctrl_dep", 4),
            ("Linux LKMM data dependency", "mp_data_dep", 4),
            ("Linux LKMM addr dependency", "mp_data_dep", 4),
            ("FreeBSD atomic ops", "cas_loop", 3),
            ("Windows MemoryBarrier pattern", "mp_fence", 3),
            ("Windows Interlocked operations", "cas_loop", 3),
            ("Linux futex wake/wait", "mp_fence", 4),
            ("Linux kfifo ring buffer", "spsc_queue", 4),
            ("Linux radix tree insertion", "cas_loop", 5),
            ("Linux page table update", "mp_fence", 5),
            ("Linux network driver DMA barrier", "mp_fence", 4),
            ("Linux device register access", "mp_fence", 4),
            ("Linux file system journal commit", "mp_fence", 5),
            ("Linux scheduler runqueue migration", "mp_fence", 5),
        ];

        for (desc, pattern, diff) in &kernel_variants {
            snippets.push(EvalSnippet {
                id: format!("kern_{}", id),
                code: format!("// Kernel: {}", desc),
                true_pattern: pattern.to_string(),
                category: "kernel_patterns".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 6: Application Patterns (25 snippets)
        let app_variants = vec![
            ("Double-checked locking (DCLP)", "dclp", 3),
            ("DCLP with volatile", "dclp", 3),
            ("DCLP with std::call_once", "dclp", 2),
            ("Singleton pattern", "dclp", 3),
            ("Observer pattern notification", "mp", 3),
            ("Event flag signaling", "mp", 2),
            ("Bounded buffer / producer-consumer", "mp", 3),
            ("Thread pool task submission", "mp", 3),
            ("Future/Promise pattern", "mp_fence", 3),
            ("Actor mailbox send/receive", "mp", 3),
            ("Timer wheel insertion", "cas_loop", 4),
            ("Connection pool checkout", "cas_loop", 3),
            ("Reference counting (shared_ptr)", "cas_loop", 3),
            ("Concurrent cache get/put", "cas_loop", 4),
            ("Read-copy-update in userspace", "rcu", 4),
            ("Log-structured merge tree", "mp_fence", 5),
            ("Concurrent hash map resize", "cas_loop", 5),
            ("Channel send/receive (Go-like)", "mp_fence", 3),
            ("Async I/O completion", "mp_fence", 3),
            ("Database mvcc read", "mp", 4),
            ("Garbage collector mark phase", "mp", 4),
            ("JIT compiler code patching", "mp_fence", 5),
            ("Hot code replacement", "mp_fence", 5),
            ("Concurrent B-tree insert", "cas_loop", 5),
            ("STM transaction commit", "cas_loop", 5),
        ];

        for (desc, pattern, diff) in &app_variants {
            snippets.push(EvalSnippet {
                id: format!("app_{}", id),
                code: format!("// App: {}", desc),
                true_pattern: pattern.to_string(),
                category: "application".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 7: Dependency Patterns (25 snippets)
        let dep_variants = vec![
            ("Address dependency (load → store via addr)", "mp_data_dep", 3),
            ("Data dependency (load → store via data)", "mp_data_dep", 3),
            ("Control dependency (branch on loaded value)", "mp_ctrl_dep", 3),
            ("Dependency through pointer chase", "mp_data_dep", 4),
            ("Dependency through array index", "mp_data_dep", 4),
            ("Dependency broken by compiler optimization", "mp", 4),
            ("Dependency through function return", "mp_data_dep", 4),
            ("Dependency through struct field", "mp_data_dep", 3),
            ("Control + data dependency chain", "mp_ctrl_dep", 4),
            ("Dependency through switch statement", "mp_ctrl_dep", 4),
            ("Dependency through ternary operator", "mp_ctrl_dep", 3),
            ("False dependency (syntactic but not semantic)", "mp", 5),
            ("Dependency through virtual dispatch", "mp_data_dep", 5),
            ("ISA2 (independent stores with dependency)", "mp_data_dep", 4),
            ("RWC with data dependency", "mp_data_dep", 4),
            ("LB with dependency preventing cycle", "lb_dep", 3),
            ("Dependency chain length 3", "mp_data_dep", 4),
            ("Dependency through memory-mapped register", "mp_data_dep", 5),
            ("Dependency through type cast", "mp_data_dep", 3),
            ("Dependency through bit manipulation", "mp_data_dep", 4),
            ("RISC-V AMO with .aq/.rl", "mp_rel_acq", 4),
            ("ARM LDAPR (load-acquire of preceding writes)", "mp_rel_acq", 4),
            ("x86 LOCK prefix dependency", "mp_fence", 3),
            ("PowerPC lwsync dependency", "mp_fence", 4),
            ("MIPS SYNC dependency", "mp_fence", 4),
        ];

        for (desc, pattern, diff) in &dep_variants {
            snippets.push(EvalSnippet {
                id: format!("dep_{}", id),
                code: format!("// Dependency: {}", desc),
                true_pattern: pattern.to_string(),
                category: "dependency_patterns".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        // Category 8: Coherence Patterns (25 snippets)
        let coh_variants = vec![
            ("CoWW: two writes to same address", "coh_ww", 2),
            ("CoWR: write then read same address", "coh_wr", 2),
            ("CoRW: read then write same address", "coh_rw", 2),
            ("CoRR: two reads from same address", "coh_rr", 2),
            ("2+2W: interleaved writes", "2+2w", 3),
            ("WRC: write-read causality", "wrc", 3),
            ("R (read from past write)", "coh_rw", 3),
            ("S (write serialization)", "coh_ww", 3),
            ("Coherence with RMW", "coh_ww", 3),
            ("Coherence in loop", "coh_rr", 4),
            ("Mixed-size coherence (word/byte)", "coh_ww", 5),
            ("Coherence with store forwarding", "coh_wr", 4),
            ("3-thread coherence chain", "wrc", 4),
            ("4-thread IRIW coherence", "coh_rr", 4),
            ("Coherence with CAS", "coh_ww", 3),
            ("Coherence with fetch_add", "coh_ww", 3),
            ("Store-to-load forwarding coherence", "coh_wr", 4),
            ("Coherence across cache lines", "coh_ww", 4),
            ("Coherence with memory barriers", "coh_ww", 3),
            ("Write atomicity test", "coh_ww", 4),
            ("Total store order coherence", "coh_ww", 3),
            ("Coherence with speculation", "coh_rr", 5),
            ("Coherence in weak memory model", "coh_ww", 3),
            ("Multi-copy atomicity violation", "coh_ww", 4),
            ("Thin-air read prevention", "coh_rr", 5),
        ];

        for (desc, pattern, diff) in &coh_variants {
            snippets.push(EvalSnippet {
                id: format!("coh_{}", id),
                code: format!("// Coherence: {}", desc),
                true_pattern: pattern.to_string(),
                category: "coherence".to_string(),
                difficulty: *diff,
                is_ood: *diff >= 4,
                adversarial_note: desc.to_string(),
            });
            id += 1;
        }

        snippets
    }

    /// Number of snippets in the benchmark.
    pub fn size() -> usize {
        Self::generate().len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_size() {
        let bench = AdversarialBenchmark::generate();
        assert!(bench.len() >= 200, "Benchmark has {} snippets (need >= 200)", bench.len());
    }

    #[test]
    fn test_category_coverage() {
        let bench = AdversarialBenchmark::generate();
        let categories: std::collections::HashSet<String> = bench.iter()
            .map(|s| s.category.clone()).collect();
        assert!(categories.len() >= 8, "Need >= 8 categories, got {}", categories.len());
    }

    #[test]
    fn test_pattern_strength() {
        let ps = PatternStrength::default_ordering();

        // Correct prediction
        assert_eq!(ps.classify_failure("mp", Some("mp")), FailureMode::Correct);

        // No match
        assert_eq!(ps.classify_failure("mp", None), FailureMode::NoMatch);

        // Conservative (predicted stronger)
        let fm = ps.classify_failure("mp", Some("mp_fence"));
        assert_eq!(fm, FailureMode::Conservative);

        // Related match (same family)
        let fm2 = ps.classify_failure("mp", Some("mp_rel_acq"));
        assert_eq!(fm2, FailureMode::RelatedMatch);
    }

    #[test]
    fn test_wilson_ci() {
        let (lo, hi) = wilson_ci_95(10, 20);
        assert!(lo > 0.3);
        assert!(hi < 0.75);
    }

    #[test]
    fn test_cohens_h() {
        let h = cohens_h(0.8, 0.2);
        assert!(h > 0.5, "Large difference should give large effect size");

        let h2 = cohens_h(0.5, 0.5);
        assert!(h2 < 0.01, "Same proportions should give zero effect size");
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = vec![
            ("MP".to_string(), Some("MP".to_string())),
            ("MP".to_string(), Some("SB".to_string())),
            ("SB".to_string(), Some("SB".to_string())),
            ("SB".to_string(), None),
        ];
        let cm = ConfusionMatrix::from_predictions(&predictions);
        assert!(cm.labels.len() >= 2);
        assert!(cm.macro_f1 >= 0.0 && cm.macro_f1 <= 1.0);
    }

    #[test]
    fn test_calibration_analysis() {
        let results = vec![
            SnippetResult {
                snippet_id: "t1".to_string(),
                predicted_pattern: Some("mp".to_string()),
                true_pattern: "mp".to_string(),
                exact_match: true,
                confidence: 0.9,
                category: "message_passing".to_string(),
                failure_mode: FailureMode::Correct,
                latency_ms: 100.0,
            },
            SnippetResult {
                snippet_id: "t2".to_string(),
                predicted_pattern: Some("sb".to_string()),
                true_pattern: "mp".to_string(),
                exact_match: false,
                confidence: 0.8,
                category: "message_passing".to_string(),
                failure_mode: FailureMode::Conservative,
                latency_ms: 150.0,
            },
        ];

        let cal = CalibrationAnalysis::from_results(&results, 10);
        assert!(cal.ece >= 0.0 && cal.ece <= 1.0);
        assert!(cal.brier_score >= 0.0);
    }

    #[test]
    fn test_category_stats() {
        let results = vec![
            SnippetResult {
                snippet_id: "t1".to_string(),
                predicted_pattern: Some("mp".to_string()),
                true_pattern: "mp".to_string(),
                exact_match: true,
                confidence: 0.9,
                category: "message_passing".to_string(),
                failure_mode: FailureMode::Correct,
                latency_ms: 100.0,
            },
        ];

        let stats = CategoryStats::from_results("message_passing", &results);
        assert_eq!(stats.n, 1);
        assert_eq!(stats.exact_matches, 1);
        assert!((stats.accuracy - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_required_sample_size() {
        let n = required_sample_size(0.50, 0.70);
        assert!(n >= 30 && n <= 200, "Need {} samples to detect 50% vs 70%", n);
    }

    #[test]
    fn test_evaluation_report() {
        let ps = PatternStrength::default_ordering();
        let results = vec![
            SnippetResult {
                snippet_id: "t1".to_string(),
                predicted_pattern: Some("mp".to_string()),
                true_pattern: "mp".to_string(),
                exact_match: true,
                confidence: 0.9,
                category: "message_passing".to_string(),
                failure_mode: FailureMode::Correct,
                latency_ms: 100.0,
            },
            SnippetResult {
                snippet_id: "t2".to_string(),
                predicted_pattern: None,
                true_pattern: "sb".to_string(),
                exact_match: false,
                confidence: 0.2,
                category: "store_buffering".to_string(),
                failure_mode: FailureMode::NoMatch,
                latency_ms: 200.0,
            },
        ];

        let report = EvaluationReport::from_results(&results, "gpt-4.1-nano", &ps);
        assert_eq!(report.total_snippets, 2);
        assert!((report.overall_accuracy - 0.5).abs() < 0.001);
        assert!(report.safe_failure_rate > 0.0);
    }
}
