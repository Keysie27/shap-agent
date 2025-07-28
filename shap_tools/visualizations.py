import matplotlib.pyplot as plt
import shap
import numpy as np

class ShapVisualizer:
    @staticmethod
    def create_all_plots(shap_values, data, waterfall_shap, waterfall_input, feature_names=None, waterfall_base_value=None):
        """Generate all SHAP plots with proper figure handling"""
        if feature_names is None and hasattr(data, 'columns'):
            feature_names = data.columns.tolist()

        if waterfall_base_value is None:
            waterfall_base_value = 0
        
        # Convert to numpy arrays if needed
        shap_array = np.array(shap_values)
        data_array = data.values if hasattr(data, 'values') else np.array(data)

        # Create figures with explicit handling
        figures = {
            'summary': ShapVisualizer._create_summary_plot(shap_array, data_array, feature_names),
            'importance': ShapVisualizer._create_importance_plot(shap_array, data_array, feature_names),
            'waterfall': ShapVisualizer._create_waterfall_plot(waterfall_shap[0], waterfall_base_value, waterfall_input.values[0], feature_names)
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
    def _create_dependence_plots(shap_values, data, feature_names, feature_index):
        plt.figure(figsize=(3, 2))
        shap.dependence_plot(
            feature_index,
            shap_values,
            data,
            feature_names=feature_names,
            show=False
        )
        fig = plt.gcf()  
        plt.close(fig)  
        return fig


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
    
    @staticmethod
    def _create_waterfall_plot(shap_values_single, base_value, data_single, feature_names):
        explanation = shap.Explanation(
            values=shap_values_single,
            base_values=base_value,
            data=data_single,
            feature_names=feature_names
        )
        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(explanation, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        return fig