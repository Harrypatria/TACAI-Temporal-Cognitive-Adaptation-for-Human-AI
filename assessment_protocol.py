"""
REI-T (Temporal) Assessment Protocol Implementation
==================================================

A comprehensive implementation of the Rational Experiential Inventory - Temporal
assessment protocol for measuring individual differences in temporal processing styles.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class REITAssessment:
    """
    Rational Experiential Inventory - Temporal (REI-T) Assessment Implementation
    
    This class provides comprehensive functionality for administering, scoring,
    and analyzing the REI-T temporal processing assessment.
    """
    
    def __init__(self):
        """Initialize the REI-T Assessment with item mappings and scoring protocols."""
        
        # Define item ranges for each subscale
        self.item_ranges = {
            'analytical': {
                'time_time': (1, 8),      # Items 1-8
                'time_event': (9, 15),    # Items 9-15
                'event_event': (16, 22)   # Items 16-22
            },
            'experiential': {
                'time_time': (23, 30),    # Items 23-30
                'time_event': (31, 37),   # Items 31-37
                'event_event': (38, 44)   # Items 38-44
            }
        }
        
        # Sample items for demonstration (abbreviated versions)
        self.sample_items = {
            'analytical': {
                'time_time': [
                    "When planning projects with multiple phases, I prefer to calculate exact durations for each phase",
                    "I like to create detailed timeline charts showing precise start and end times for activities",
                    "When coordinating overlapping activities, I systematically analyze temporal interval relationships"
                ],
                'time_event': [
                    "I prefer to schedule events based on systematic analysis of optimal timing windows",
                    "When coordinating events, I like to map out explicit relationships between timing and outcomes",
                    "I systematically analyze how event timing affects success probability"
                ],
                'event_event': [
                    "I prefer to map out explicit causal relationships between sequential events",
                    "When managing complex projects, I create detailed dependency charts showing event relationships",
                    "I systematically analyze how delays in one event will affect subsequent events"
                ]
            },
            'experiential': {
                'time_time': [
                    "I trust my gut about when to do what as part of a 24-hour day",
                    "When juggling activities, I'd rather look at the big picture, not the clock",
                    "I like a relaxed schedule that can fit the flow of things"
                ],
                'time_event': [
                    "I schedule events based on intuition of timing for events",
                    "When planning events, I go with what feels appropriate for the situation",
                    "When timing counts, I go off my gut feel for timing"
                ],
                'event_event': [
                    "I use pattern recognition to make connections between events",
                    "I concentrate on the organic order, rather than syntactic dependencies",
                    "I trust my gut feel how events affect each other"
                ]
            }
        }
    
    def score_assessment(self, responses: Dict[int, int]) -> Dict[str, float]:
        """
        Score REI-T assessment responses.
        
        Args:
            responses: Dictionary mapping item numbers (1-44) to ratings (1-7)
            
        Returns:
            Dictionary containing all computed scores
        """
        if len(responses) != 44:
            raise ValueError("REI-T requires responses to all 44 items")
        
        if not all(1 <= item <= 44 for item in responses.keys()):
            raise ValueError("Item numbers must be between 1 and 44")
            
        if not all(1 <= rating <= 7 for rating in responses.values()):
            raise ValueError("Ratings must be between 1 and 7")
        
        scores = {}
        
        # Calculate analytical subscale scores
        analytical_items = []
        for start, end in self.item_ranges['analytical'].values():
            analytical_items.extend(range(start, end + 1))
        
        rei_t_a = np.mean([responses[item] for item in analytical_items])
        scores['REI_T_A'] = rei_t_a
        
        # Calculate experiential subscale scores
        experiential_items = []
        for start, end in self.item_ranges['experiential'].values():
            experiential_items.extend(range(start, end + 1))
        
        rei_t_e = np.mean([responses[item] for item in experiential_items])
        scores['REI_T_E'] = rei_t_e
        
        # Calculate Temporal Processing Difference
        scores['TPD'] = rei_t_a - rei_t_e
        
        # Determine cognitive style classification
        if rei_t_a > rei_t_e + 0.5:
            scores['temporal_style'] = 'Analytical'
        elif rei_t_e > rei_t_a + 0.5:
            scores['temporal_style'] = 'Experiential'
        else:
            scores['temporal_style'] = 'Versatile'
        
        # Calculate detailed subscale scores
        for scale in ['analytical', 'experiential']:
            for reasoning_type in ['time_time', 'time_event', 'event_event']:
                start, end = self.item_ranges[scale][reasoning_type]
                items = list(range(start, end + 1))
                subscale_score = np.mean([responses[item] for item in items])
                scores[f'{reasoning_type}_{scale}'] = subscale_score
        
        return scores


class TemporalBehaviorAnalyzer:
    """
    Analyzes temporal behavioral patterns from user interaction data.
    """
    
    def __init__(self):
        """Initialize the temporal behavior analyzer."""
        self.temporal_features = {}
    
    def analyze_inter_prompt_intervals(self, timestamps: List[float], user_id: str) -> Dict[str, float]:
        """
        Analyze inter-prompt interval patterns.
        
        Args:
            timestamps: List of interaction timestamps
            user_id: Identifier for the user
            
        Returns:
            Dictionary of temporal pattern metrics
        """
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 timestamps for interval analysis")
        
        # Calculate inter-prompt intervals
        ipis = np.diff(timestamps)
        
        # Basic statistics
        mu_ipi = np.mean(ipis)
        sigma_ipi = np.std(ipis, ddof=1) if len(ipis) > 1 else 0
        
        # Temporal Regularity Index
        tri = 1 - (sigma_ipi / mu_ipi) if mu_ipi > 0 else 0
        
        # Coefficient of Variation
        cv = sigma_ipi / mu_ipi if mu_ipi > 0 else 0
        
        metrics = {
            'mean_ipi': mu_ipi,
            'std_ipi': sigma_ipi,
            'temporal_regularity_index': tri,
            'coefficient_of_variation': cv,
            'n_intervals': len(ipis)
        }
        
        return metrics
    
    def analyze_temporal_rhythm(self, ipi_sequence: np.ndarray) -> Dict[str, float]:
        """
        Analyze temporal rhythm using Discrete Fourier Transform.
        
        Args:
            ipi_sequence: Array of inter-prompt intervals
            
        Returns:
            Dictionary of rhythm analysis metrics
        """
        if len(ipi_sequence) < 4:
            return {
                'dominant_frequency': 0,
                'temporal_predictability': 0,
                'spectral_entropy': 0,
                'rhythm_strength': 0
            }
        
        # Apply DFT
        fft_result = fft(ipi_sequence)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Normalize power spectrum
        total_power = np.sum(power_spectrum)
        if total_power > 0:
            normalized_spectrum = power_spectrum / total_power
        else:
            normalized_spectrum = power_spectrum
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
        
        # Calculate temporal predictability
        max_power = np.max(power_spectrum)
        temporal_predictability = max_power / total_power if total_power > 0 else 0
        
        # Calculate spectral entropy
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        p_normalized = normalized_spectrum + epsilon
        spectral_entropy = -np.sum(p_normalized * np.log2(p_normalized))
        
        # Rhythm strength (dominant frequency power relative to total)
        rhythm_strength = power_spectrum[dominant_freq_idx] / total_power if total_power > 0 else 0
        
        return {
            'dominant_frequency': dominant_freq_idx,
            'temporal_predictability': temporal_predictability,
            'spectral_entropy': spectral_entropy,
            'rhythm_strength': rhythm_strength
        }
    
    def calculate_cognitive_load(self, nasa_tlx: List[float], response_delays: List[float], 
                               error_rates: List[float], revision_frequencies: List[float],
                               weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate cognitive load temporal distribution.
        
        Args:
            nasa_tlx: NASA Task Load Index scores over time
            response_delays: Response delay measurements
            error_rates: Error rate measurements
            revision_frequencies: Revision frequency measurements
            weights: Optional weights for combining metrics
            
        Returns:
            Dictionary of cognitive load metrics
        """
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
        
        if len(weights) != 4:
            raise ValueError("Must provide exactly 4 weights")
        
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        # Ensure all arrays have the same length
        min_length = min(len(nasa_tlx), len(response_delays), 
                        len(error_rates), len(revision_frequencies))
        
        nasa_tlx = np.array(nasa_tlx[:min_length])
        response_delays = np.array(response_delays[:min_length])
        error_rates = np.array(error_rates[:min_length])
        revision_frequencies = np.array(revision_frequencies[:min_length])
        
        # Calculate temporal cognitive load
        cl_temporal = (weights[0] * nasa_tlx + 
                      weights[1] * response_delays + 
                      weights[2] * error_rates + 
                      weights[3] * revision_frequencies)
        
        # Calculate derived metrics
        cl_integral = np.mean(cl_temporal)
        cl_variance = np.var(cl_temporal)
        peak_load = np.max(cl_temporal)
        load_smoothness = 1 - (np.std(cl_temporal) / np.mean(cl_temporal)) if np.mean(cl_temporal) > 0 else 0
        
        return {
            'cl_integral': cl_integral,
            'cl_variance': cl_variance,
            'peak_load': peak_load,
            'load_smoothness': load_smoothness,
            'cl_temporal_series': cl_temporal.tolist()
        }


class TemporalAdaptationEvaluator:
    """
    Evaluates effectiveness of temporal adaptations.
    """
    
    def __init__(self):
        """Initialize the temporal adaptation evaluator."""
        pass
    
    def calculate_tae(self, performance_improvement: float, trust_enhancement: float,
                     cognitive_load_reduction: float, user_satisfaction: float,
                     weights: Optional[List[float]] = None) -> float:
        """
        Calculate Temporal Adaptation Effectiveness (TAE).
        
        Args:
            performance_improvement: Performance improvement metric
            trust_enhancement: Trust enhancement metric
            cognitive_load_reduction: Cognitive load reduction metric
            user_satisfaction: User satisfaction metric
            weights: Optional weights for combining metrics
            
        Returns:
            TAE score
        """
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
        
        if len(weights) != 4:
            raise ValueError("Must provide exactly 4 weights")
        
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        tae = (weights[0] * performance_improvement +
               weights[1] * trust_enhancement +
               weights[2] * cognitive_load_reduction +
               weights[3] * user_satisfaction)
        
        return tae
    
    def calculate_overall_benefit(self, tae_score: float, sustainability_factor: float,
                                generalization_factor: float) -> float:
        """
        Calculate overall benefit of temporal adaptation.
        
        Args:
            tae_score: Temporal Adaptation Effectiveness score
            sustainability_factor: How long-lasting the benefits are
            generalization_factor: How broadly the benefits apply
            
        Returns:
            Overall benefit score
        """
        return tae_score * sustainability_factor * generalization_factor


class TemporalMLClassifier:
    """
    Machine learning classifier for temporal cognitive styles.
    """
    
    def __init__(self):
        """Initialize the ML classifier."""
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
    
    def prepare_temporal_features(self, temporal_data: Dict[str, List[float]]) -> np.ndarray:
        """
        Prepare temporal feature vector for classification.
        
        Args:
            temporal_data: Dictionary containing temporal metrics
            
        Returns:
            Feature vector array
        """
        expected_features = [
            'mean_ipi', 'std_ipi', 'temporal_regularity_index', 
            'burst_frequency', 'sequential_index', 'coordination_quality',
            'adaptation_speed', 'trust_trajectory', 'rhythm_strength',
            'spectral_entropy', 'peak_load', 'load_smoothness'
        ]
        
        features = []
        for feature in expected_features:
            if feature in temporal_data:
                if isinstance(temporal_data[feature], list):
                    features.append(np.mean(temporal_data[feature]))
                else:
                    features.append(temporal_data[feature])
            else:
                features.append(0.0)  # Default value for missing features
        
        return np.array(features).reshape(1, -1)
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train ensemble classification models.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_train_scaled = self.scalers['main'].fit_transform(X_train)
        X_test_scaled = self.scalers['main'].transform(X_test)
        
        # Train individual models
        self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.models['svm'] = SVC(probability=True, random_state=random_state)
        
        self.models['rf'].fit(X_train_scaled, y_train)
        self.models['svm'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        performance = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            performance[f'{name}_accuracy'] = accuracy
        
        self.is_fitted = True
        return performance
    
    def predict_ensemble(self, X: np.ndarray, weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            weights: Optional weights for ensemble models
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        if weights is None:
            weights = [0.5, 0.5]  # Equal weights for RF and SVM
        
        X_scaled = self.scalers['main'].transform(X)
        
        # Get probability predictions from each model
        rf_probs = self.models['rf'].predict_proba(X_scaled)
        svm_probs = self.models['svm'].predict_proba(X_scaled)
        
        # Combine predictions using weights
        ensemble_probs = weights[0] * rf_probs + weights[1] * svm_probs
        
        # Return class with highest probability
        return np.argmax(ensemble_probs, axis=1)


class REITVisualization:
    """
    Visualization tools for REI-T assessment results and temporal patterns.
    """
    
    def __init__(self):
        """Initialize visualization tools."""
        plt.style.use('seaborn-v0_8')
    
    def plot_assessment_profile(self, scores: Dict[str, float], save_path: Optional[str] = None):
        """
        Create a radar plot of REI-T assessment scores.
        
        Args:
            scores: Dictionary of assessment scores
            save_path: Optional path to save the plot
        """
        # Extract subscale scores for radar plot
        subscales = ['time_time_analytical', 'time_event_analytical', 'event_event_analytical',
                    'time_time_experiential', 'time_event_experiential', 'event_event_experiential']
        
        values = [scores.get(subscale, 0) for subscale in subscales]
        labels = ['TT-A', 'TE-A', 'EE-A', 'TT-E', 'TE-E', 'EE-E']
        
        # Create radar plot
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='Scores')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 7)
        ax.set_title(f'REI-T Profile: {scores.get("temporal_style", "Unknown")} Style', 
                    size=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        # Add score annotations
        textstr = f"""REI-T-A: {scores.get('REI_T_A', 0):.2f}
REI-T-E: {scores.get('REI_T_E', 0):.2f}
TPD: {scores.get('TPD', 0):.2f}"""
        
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_temporal_patterns(self, timestamps: List[float], save_path: Optional[str] = None):
        """
        Visualize temporal interaction patterns.
        
        Args:
            timestamps: List of interaction timestamps
            save_path: Optional path to save the plot
        """
        if len(timestamps) < 2:
            print("Need at least 2 timestamps for temporal pattern visualization")
            return
        
        ipis = np.diff(timestamps)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Inter-prompt intervals over time
        ax1.plot(range(len(ipis)), ipis, 'b-', marker='o', alpha=0.7)
        ax1.set_title('Inter-Prompt Intervals Over Time')
        ax1.set_xlabel('Interaction Number')
        ax1.set_ylabel('Time Interval (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of intervals
        ax2.hist(ipis, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Distribution of Inter-Prompt Intervals')
        ax2.set_xlabel('Time Interval (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative interaction timeline
        cumulative_time = np.cumsum([0] + list(ipis))
        ax3.plot(cumulative_time, range(len(cumulative_time)), 's-', alpha=0.7, color='red')
        ax3.set_title('Cumulative Interaction Timeline')
        ax3.set_xlabel('Cumulative Time (seconds)')
        ax3.set_ylabel('Interaction Count')
        ax3.grid(True, alpha=0.3)
        
        # Rhythm analysis (if sufficient data)
        if len(ipis) >= 4:
            analyzer = TemporalBehaviorAnalyzer()
            rhythm_metrics = analyzer.analyze_temporal_rhythm(ipis)
            
            # Simple frequency domain visualization
            freqs = np.fft.fftfreq(len(ipis))
            fft_result = np.fft.fft(ipis)
            power = np.abs(fft_result) ** 2
            
            ax4.plot(freqs[:len(freqs)//2], power[:len(power)//2])
            ax4.set_title('Temporal Rhythm Analysis (Power Spectrum)')
            ax4.set_xlabel('Frequency')
            ax4.set_ylabel('Power')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor rhythm analysis', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Temporal Rhythm Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Example usage and demonstration
def demonstrate_rei_t():
    """Demonstrate the REI-T assessment protocol with example data."""
    
    print("REI-T Assessment Protocol Demonstration")
    print("=" * 50)
    
    # Initialize assessment
    rei_t = REITAssessment()
    
    # Example responses (simulated data)
    example_responses = {}
    
    # Simulate analytical-leaning responses
    for i in range(1, 23):  # Analytical items
        example_responses[i] = np.random.randint(5, 8)  # Higher scores
    
    for i in range(23, 45):  # Experiential items
        example_responses[i] = np.random.randint(2, 5)  # Lower scores
    
    # Score the assessment
    scores = rei_t.score_assessment(example_responses)
    
    print("\nAssessment Scores:")
    print("-" * 20)
    for key, value in scores.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Temporal behavior analysis
    print("\nTemporal Behavior Analysis:")
    print("-" * 30)
    
    # Simulate interaction timestamps
    base_time = 0
    timestamps = []
    for i in range(20):
        # Simulate variable intervals with some regularity
        interval = np.random.normal(30, 10)  # Mean 30 seconds, std 10
        base_time += max(interval, 5)  # Minimum 5 seconds
        timestamps.append(base_time)
    
    analyzer = TemporalBehaviorAnalyzer()
    
    # Analyze intervals
    interval_metrics = analyzer.analyze_inter_prompt_intervals(timestamps, "user_001")
    print(f"Mean IPI: {interval_metrics['mean_ipi']:.2f} seconds")
    print(f"Temporal Regularity: {interval_metrics['temporal_regularity_index']:.3f}")
    print(f"Coefficient of Variation: {interval_metrics['coefficient_of_variation']:.3f}")
    
    # Analyze rhythm
    ipis = np.diff(timestamps)
    rhythm_metrics = analyzer.analyze_temporal_rhythm(ipis)
    print(f"Spectral Entropy: {rhythm_metrics['spectral_entropy']:.3f}")
    print(f"Rhythm Strength: {rhythm_metrics['rhythm_strength']:.3f}")
    
    # Cognitive load analysis
    print("\nCognitive Load Analysis:")
    print("-" * 25)
    
    # Simulate cognitive load data
    n_points = len(timestamps)
    nasa_tlx = np.random.normal(50, 15, n_points)
    response_delays = np.random.exponential(2, n_points)
    error_rates = np.random.beta(2, 8, n_points)
    revision_frequencies = np.random.poisson(1, n_points)
    
    cl_metrics = analyzer.calculate_cognitive_load(
        nasa_tlx.tolist(), response_delays.tolist(),
        error_rates.tolist(), revision_frequencies.tolist()
    )
    
    print(f"Integral Load: {cl_metrics['cl_integral']:.3f}")
    print(f"Peak Load: {cl_metrics['peak_load']:.3f}")
    print(f"Load Smoothness: {cl_metrics['load_smoothness']:.3f}")
    
    # Temporal Adaptation Effectiveness
    print("\nTemporal Adaptation Effectiveness:")
    print("-" * 35)
    
    evaluator = TemporalAdaptationEvaluator()
    tae_score = evaluator.calculate_tae(0.8, 0.7, 0.6, 0.9)
    overall_benefit = evaluator.calculate_overall_benefit(tae_score, 0.85, 0.75)
    
    print(f"TAE Score: {tae_score:.3f}")
    print(f"Overall Benefit: {overall_benefit:.3f}")
    
    # Visualization
    print("\nGenerating visualizations...")
    visualizer = REITVisualization()
    
    # Plot assessment profile
    visualizer.plot_assessment_profile(scores)
    
    # Plot temporal patterns
    visualizer.plot_temporal_patterns(timestamps)
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    demonstrate_rei_t()
