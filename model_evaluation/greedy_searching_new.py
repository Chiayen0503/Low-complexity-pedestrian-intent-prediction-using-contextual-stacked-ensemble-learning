import torch
from torchmetrics.functional import precision_recall
from torchmetrics import F1Score, AUROC
import numpy as np

class ThresholdGreedySearcher:
    """
    A greedy search class to find the best threshold for various metrics.
    Preserves all scores from all metrics across all thresholds.
    """
    
    def __init__(self, threshold_range=(0.0, 1.0), step=0.01, device='cuda'):
        """
        Args:
            threshold_range: tuple of (min, max) threshold values
            step: step size for threshold search
            device: torch device ('cuda' or 'cpu')
        """
        self.threshold_range = threshold_range
        self.step = step
        self.device = torch.device(device)
        
        # Generate threshold list
        self.thresholds = np.arange(
            threshold_range[0], 
            threshold_range[1] + step, 
            step
        ).tolist()
        
        # Storage for all results
        self.results = {
            'thresholds': [],
            'accuracy': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        # Best results
        self.best_results = {}
        
    def threshold_based_acc(self, probs, targets, threshold):
        """Calculate threshold-based accuracy"""
        preds = (probs >= threshold).int()
        acc = (preds == targets).float().mean().item()
        return acc
    
    def compute_all_metrics(self, probs, targets, threshold):
        """Compute all metrics for a given threshold"""
        # Ensure correct dtypes and device
        probs = probs.to(self.device)
        targets = targets.long().to(self.device)  # Convert to long
        
        # Precision and Recall
        p, r = precision_recall(
            probs, targets, 
            average='macro', 
            threshold=threshold, 
            num_classes=1
        )
        p = p.item()
        r = r.item()
        
        # F1 Score
        f1_func = F1Score(task="binary", threshold=threshold, average='macro').to(self.device)
        f1 = f1_func(probs, targets).item()
        
        # AUROC
        auroc = AUROC(task="binary", threshold=threshold)
        auc = auroc(probs, targets).item()
        
        # Accuracy
        acc = self.threshold_based_acc(probs, targets, threshold)
        
        return {
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'precision': p,
            'recall': r
        }
    
    def search(self, probs, targets, optimize_metric='f1', verbose=False):
        """
        Perform greedy search to find best threshold.
        
        Args:
            probs: predicted probabilities (torch.Tensor)
            targets: ground truth labels (torch.Tensor)
            optimize_metric: metric to optimize ('f1', 'accuracy', 'auc', 'precision', 'recall')
            verbose: whether to print progress
            
        Returns:
            best_threshold: threshold that gives best score
            best_score: best score achieved
        """
        if optimize_metric not in ['f1', 'accuracy', 'auc', 'precision', 'recall']:
            raise ValueError(f"Invalid metric: {optimize_metric}. Choose from: f1, accuracy, auc, precision, recall")
        
        best_score = -float('inf')
        best_threshold = None
        
        # Clear previous results
        self.results = {
            'thresholds': [],
            'accuracy': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        
        print(f"Searching for best threshold optimizing {optimize_metric}...")
        print(f"Threshold range: {self.threshold_range}, Step: {self.step}")
        print("-" * 60)
        
        for threshold in self.thresholds:
            # Compute all metrics
            metrics = self.compute_all_metrics(probs, targets, threshold)
            
            # Store results
            self.results['thresholds'].append(threshold)
            for metric_name, value in metrics.items():
                self.results[metric_name].append(value)
            
            # Check if this is the best score
            current_score = metrics[optimize_metric]
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
                self.best_results = {
                    'threshold': best_threshold,
                    'optimized_metric': optimize_metric,
                    **metrics
                }
            
            if verbose:
                print(f"Threshold: {threshold:.3f} | {optimize_metric}: {current_score:.4f}")
        
        print("-" * 60)
        print(f"\n✓ Search completed!")
        print(f"Best threshold: {best_threshold:.3f}")
        print(f"Best {optimize_metric}: {best_score:.4f}")
        print("\nAll metrics at best threshold:")
        for metric_name, value in self.best_results.items():
            if metric_name not in ['threshold', 'optimized_metric']:
                print(f"  {metric_name}: {value:.4f}")
        
        return best_threshold, best_score
    
    def get_best_results(self):
        """Return dictionary of best results"""
        return self.best_results
    
    def get_all_results(self):
        """Return dictionary of all results across all thresholds"""
        return self.results
    
    def get_metrics_at_threshold(self, threshold):
        """Get all metrics at a specific threshold"""
        if threshold not in self.results['thresholds']:
            print(f"Warning: Threshold {threshold} not in searched thresholds")
            return None
        
        idx = self.results['thresholds'].index(threshold)
        return {
            'threshold': threshold,
            'accuracy': self.results['accuracy'][idx],
            'auc': self.results['auc'][idx],
            'f1': self.results['f1'][idx],
            'precision': self.results['precision'][idx],
            'recall': self.results['recall'][idx]
        }
    
    def plot_results(self, metrics_to_plot=None, figsize=(12, 6)):
        """
        Plot metrics across thresholds (requires matplotlib)
        
        Args:
            metrics_to_plot: list of metrics to plot (default: all)
            figsize: figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot results.")
            return
        
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall', 'auc']
        
        plt.figure(figsize=figsize)
        
        for metric in metrics_to_plot:
            if metric in self.results:
                plt.plot(self.results['thresholds'], 
                        self.results[metric], 
                        label=metric.capitalize(), 
                        marker='o', 
                        markersize=2)
        
        # Mark best threshold
        if self.best_results:
            plt.axvline(x=self.best_results['threshold'], 
                       color='red', 
                       linestyle='--', 
                       label=f"Best threshold ({self.best_results['threshold']:.3f})")
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Sample data
    torch.manual_seed(42)
    probs = torch.rand(1000)
    targets = (torch.rand(1000) > 0.5).float()  # Can be float initially
    
    # Initialize searcher
    searcher = ThresholdGreedySearcher(
        threshold_range=(0.0, 1.0),
        step=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Search for best F1 - targets will be converted to long inside
    best_threshold, best_f1 = searcher.search(
        probs, targets, 
        optimize_metric='f1',
        verbose=False
    )
    
    # Get best results
    print("\n" + "="*60)
    print("BEST RESULTS:")
    best_results = searcher.get_best_results()
    for key, value in best_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")