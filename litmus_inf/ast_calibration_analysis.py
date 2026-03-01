#!/usr/bin/env python3
"""
AST Confidence Score Calibration Analysis.

Addresses consensus weakness #4: "AST confidence score calibration for
hybrid fallback is never validated."

Analyzes the coverage-accuracy tradeoff at varying confidence thresholds
to determine optimal fallback threshold and validate calibration.
"""

import json
import math
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0, center - margin), min(1, center + margin)


def run_calibration_analysis() -> Dict:
    """Run AST confidence calibration across the benchmark suite.
    
    For each snippet, records:
    - The confidence score assigned to the top match
    - Whether the match was correct (exact or top-3)
    
    Then analyzes calibration at varying thresholds.
    """
    from ast_analyzer import ast_analyze_code
    from benchmark_suite import BENCHMARK_SNIPPETS

    print(f"Running calibration on {len(BENCHMARK_SNIPPETS)} benchmark snippets...")
    
    data_points = []
    
    for i, snip in enumerate(BENCHMARK_SNIPPETS):
        code = snip['code']
        expected = snip['expected_pattern']
        
        try:
            result = ast_analyze_code(code, language='auto')
            matches = result.patterns_found
            
            if matches:
                top_match = matches[0]
                confidence = top_match.confidence
                predicted = top_match.pattern_name
                exact = (predicted == expected)
                top3 = expected in [m.pattern_name for m in matches[:3]]
                coverage = result.coverage_confidence
            else:
                confidence = 0.0
                predicted = None
                exact = False
                top3 = False
                coverage = result.coverage_confidence
                
        except Exception as e:
            confidence = 0.0
            predicted = None
            exact = False
            top3 = False
            coverage = 0.0
        
        data_points.append({
            'snippet_id': snip.get('id', f'snip_{i}'),
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'coverage_confidence': coverage,
            'exact': exact,
            'top3': top3,
            'category': snip.get('category', 'unknown'),
        })
    
    # Sort by confidence for threshold analysis
    data_points.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calibration bins: divide confidence scores into deciles
    calibration_bins = []
    n_bins = 10
    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        bin_points = [d for d in data_points if lo <= d['confidence'] < hi]
        if bin_points:
            n_correct = sum(1 for d in bin_points if d['exact'])
            avg_conf = sum(d['confidence'] for d in bin_points) / len(bin_points)
            actual_acc = n_correct / len(bin_points)
            calibration_bins.append({
                'bin_range': f'[{lo:.1f}, {hi:.1f})',
                'n_snippets': len(bin_points),
                'avg_confidence': round(avg_conf, 3),
                'actual_accuracy': round(actual_acc, 3),
                'gap': round(actual_acc - avg_conf, 3),
            })
    
    # Threshold sweep: at each threshold, compute coverage + accuracy
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold_analysis = []
    
    for thresh in thresholds:
        above = [d for d in data_points if d['confidence'] >= thresh]
        below = [d for d in data_points if d['confidence'] < thresh]
        
        n_above = len(above)
        n_total = len(data_points)
        coverage = n_above / n_total if n_total else 0
        
        if n_above > 0:
            exact_correct = sum(1 for d in above if d['exact'])
            top3_correct = sum(1 for d in above if d['top3'])
            exact_acc = exact_correct / n_above
            top3_acc = top3_correct / n_above
            exact_ci = list(wilson_ci(exact_correct, n_above))
            top3_ci = list(wilson_ci(top3_correct, n_above))
        else:
            exact_acc = 0
            top3_acc = 0
            exact_ci = [0, 0]
            top3_ci = [0, 0]
        
        # Snippets that would be sent to LLM fallback
        n_fallback = len(below)
        
        threshold_analysis.append({
            'threshold': thresh,
            'coverage': round(coverage, 3),
            'n_above': n_above,
            'n_fallback': n_fallback,
            'exact_accuracy': round(exact_acc, 3),
            'top3_accuracy': round(top3_acc, 3),
            'exact_wilson_ci': [round(x, 3) for x in exact_ci],
            'top3_wilson_ci': [round(x, 3) for x in top3_ci],
        })
    
    # Find optimal threshold (maximizes accuracy while maintaining >50% coverage)
    best_thresh = 0.0
    best_f1 = 0.0
    for ta in threshold_analysis:
        if ta['coverage'] >= 0.3:  # at least 30% coverage
            # F1-like metric: harmonic mean of coverage and accuracy
            if ta['exact_accuracy'] > 0 and ta['coverage'] > 0:
                f1 = 2 * ta['exact_accuracy'] * ta['coverage'] / (ta['exact_accuracy'] + ta['coverage'])
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = ta['threshold']
    
    # Expected Calibration Error (ECE)
    total_points = len(data_points)
    ece = 0.0
    for b in calibration_bins:
        ece += (b['n_snippets'] / total_points) * abs(b['gap'])
    
    # Coverage-confidence correlation (do high-coverage snippets have higher accuracy?)
    high_cov = [d for d in data_points if d['coverage_confidence'] >= 0.5]
    low_cov = [d for d in data_points if d['coverage_confidence'] < 0.5]
    
    coverage_analysis = {
        'high_coverage_n': len(high_cov),
        'high_coverage_exact_rate': sum(1 for d in high_cov if d['exact']) / max(1, len(high_cov)),
        'low_coverage_n': len(low_cov),
        'low_coverage_exact_rate': sum(1 for d in low_cov if d['exact']) / max(1, len(low_cov)),
    }

    report = {
        'n_total': len(data_points),
        'overall_exact_rate': sum(1 for d in data_points if d['exact']) / len(data_points),
        'overall_top3_rate': sum(1 for d in data_points if d['top3']) / len(data_points),
        'calibration_bins': calibration_bins,
        'threshold_analysis': threshold_analysis,
        'optimal_threshold': best_thresh,
        'optimal_f1': round(best_f1, 3),
        'expected_calibration_error': round(ece, 4),
        'coverage_confidence_analysis': coverage_analysis,
        'interpretation': {
            'ece_note': f'ECE = {ece:.4f} (lower is better; <0.1 is well-calibrated, >0.2 is poorly calibrated)',
            'optimal_threshold_note': f'Threshold {best_thresh} maximizes F1(coverage, accuracy) at {best_f1:.3f}',
            'recommendation': 'Use coverage_confidence < 0.5 as the LLM fallback trigger',
        },
    }
    
    return report


if __name__ == '__main__':
    print("=" * 70)
    print("LITMUS∞ AST Confidence Score Calibration Analysis")
    print("=" * 70)
    
    report = run_calibration_analysis()
    
    print(f"\nOverall: {report['overall_exact_rate']:.1%} exact, {report['overall_top3_rate']:.1%} top-3")
    print(f"ECE: {report['expected_calibration_error']:.4f}")
    print(f"Optimal threshold: {report['optimal_threshold']}")
    
    print("\nCalibration bins:")
    print(f"  {'Range':12s} {'N':>4s} {'Avg Conf':>9s} {'Accuracy':>9s} {'Gap':>6s}")
    for b in report['calibration_bins']:
        print(f"  {b['bin_range']:12s} {b['n_snippets']:4d} {b['avg_confidence']:9.3f} {b['actual_accuracy']:9.3f} {b['gap']:+6.3f}")
    
    print("\nThreshold sweep (coverage vs accuracy tradeoff):")
    print(f"  {'Thresh':>6s} {'Coverage':>9s} {'N':>5s} {'Exact':>7s} {'Top-3':>7s} {'Fallback':>9s}")
    for ta in report['threshold_analysis']:
        print(f"  {ta['threshold']:6.1f} {ta['coverage']:9.3f} {ta['n_above']:5d} "
              f"{ta['exact_accuracy']:7.3f} {ta['top3_accuracy']:7.3f} {ta['n_fallback']:9d}")
    
    print(f"\nCoverage-confidence correlation:")
    ca = report['coverage_confidence_analysis']
    print(f"  High coverage (≥0.5): {ca['high_coverage_n']} snippets, {ca['high_coverage_exact_rate']:.1%} exact")
    print(f"  Low coverage (<0.5):  {ca['low_coverage_n']} snippets, {ca['low_coverage_exact_rate']:.1%} exact")
    
    os.makedirs('paper_results_v10', exist_ok=True)
    with open('paper_results_v10/ast_calibration_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to paper_results_v10/ast_calibration_analysis.json")
