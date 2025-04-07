import matplotlib.pyplot as plt
import shap
import numpy as np

class ShapVisualizer:
    @staticmethod
    def create_all_plots(shap_values, data, feature_names=None):
        """Generate all SHAP plots with proper figure handling"""
        if feature_names is None and hasattr(data, 'columns'):
            feature_names = data.columns.tolist()
        
        # Convert to numpy arrays if needed
        shap_array = np.array(shap_values)
        data_array = data.values if hasattr(data, 'values') else np.array(data)
        
        # Create figures with explicit handling
        figures = {
            'summary': ShapVisualizer._create_summary_plot(shap_array, data_array, feature_names),
            'dependence': ShapVisualizer._create_dependence_plots(shap_array, data_array, feature_names),
            'importance': ShapVisualizer._create_importance_plot(shap_array, data_array, feature_names)
        }
        
        return figures

    @staticmethod
    def _create_summary_plot(shap_values, data, feature_names):
        """Create beeswarm summary plot"""
        fig, _ = plt.subplots(figsize=(3, 2))
        shap.summary_plot(
            shap_values, 
            features=data,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        return fig

    @staticmethod
    def _create_dependence_plots(shap_values, data, feature_names, n_plots=3):
        """Create dependence plots for top features"""
        plots = []
        for i in range(min(n_plots, shap_values.shape[1])):
            fig, _ = plt.subplots(figsize=(3, 2))
            shap.dependence_plot(
                i,
                shap_values,
                data,
                feature_names=feature_names,
                show=False
            )
            plots.append(fig)
        return plots

    @staticmethod
    def _create_importance_plot(shap_values, data, feature_names):
        """Create feature importance bar plot"""
        fig, _ = plt.subplots(figsize=(3, 2))
        shap.summary_plot(
            shap_values,
            features=data,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=5
        )
        plt.tight_layout()
        return fig
