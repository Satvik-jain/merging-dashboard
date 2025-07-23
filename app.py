import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

class ModelVisualizationDashboard:
    def __init__(self):
        # Initialize data
        self.setup_data()
        self.colors = {
            'dare_linear': '#3B82F6',
            'dare_ties': '#10B981', 
            'linear': '#F59E0B',
            'magnitude_prune': '#EF4444',
            'ties': '#8B5CF6',
            'della': '#EC4899',
            'arcee_fusion': '#14B8A6',
            'model_stock': '#F97316'
        }
        
    def setup_data(self):
        # Merging methods data
        merging_data = [
            ("dare_linear_5_5_d5", "dare_linear", 0.5545, 0.3909, 0.6510, 0.2545),
            ("dare_ties_5_5_d5", "dare_ties", 0.5000, 0.4909, 0.6296, 0.3091),
            ("linear_1_9", "linear", 0.0183, 0.0000, 0.0000, 0.0000),
            ("magnitude_prune_5_5_d3", "magnitude_prune", 0.2182, 0.0818, 0.2225, 0.0182),
            ("ties_3_7_d5", "ties", 0.2182, 0.0455, 0.8000, 0.0364),
            ("ties_5_5_d5", "ties", 0.1182, 0.0727, 0.1252, 0.0091),
            ("ties_7_3_d5", "ties", 0.0636, 0.0091, 1.0000, 0.0091),
            ("dare_linear_3_7_d8", "dare_linear", 0.2091, 0.0455, 0.4000, 0.0182),
            ("dare_ties_2_8_d9_freq", "dare_ties", 0.6000, 0.2636, 0.8277, 0.2182),
            ("dare_ties_3_7_d8_freq", "dare_ties", 0.598112, 0.252383, 0.913718, 0.224318),
            ("ties_1_9_d8_freq", "ties", 0.0183, 0.0275, 0.0000, 0.0000),
            ("ties_2_8_d7_freq", "ties", 0.0091, 0.0182, 0.0000, 0.0000),
            ("ties_5_5_d9_freq", "ties", 0.3500, 0.1400, 0.4278, 0.0599),
            ("ties_7_3_d7_freq", "ties", 0.1364, 0.0455, 0.0000, 0.0000),
            ("ties_7_3_d10_freq", "ties", 0.1182, 0.0636, 0.1431, 0.0091),
            ("ties_9_1_d8_freq", "ties", 0.1727, 0.1182, 0.3849, 0.0455)
        ]
        
        # Model variations data
        models = ["arcee_fusion", "model_stock", "della_1_9_d5", "della_2_8_d5", "della_3_7_d5", 
                 "della_5_5_d5", "della_7_3_d7", "della_1_9_d8", "della_2_8_d7", "della_2_8_d8", 
                 "della_3_7_d8", "della_5_5_d8", "della_7_3_d5"]
        
        compilation_rate = [0.6636, 0.0545, 0.5818, 0.6364, 0.5636, 0.5545, 0.5818, 0.6111, 
                           0.6111, 0.5833, 0.6238, 0.5925, 0.5364]
        semantic_rate = [0.4000, 0.4455, 0.3364, 0.2909, 0.2455, 0.3273, 0.2909, 0.4074, 
                        0.4166, 0.4352, 0.3333, 0.3569, 0.2000]
        conditioned_p = [0.7500, 0.1021, 0.7027, 0.7500, 0.8517, 0.7222, 0.7188, 0.6590, 
                        0.6606, 0.6167, 0.7566, 0.7500, 0.6820]
        joint_p = [0.3000, 0.0455, 0.2364, 0.2182, 0.2091, 0.2364, 0.2091, 0.2685, 
                  0.2752, 0.2684, 0.2521, 0.2677, 0.1364]
        
        # Create DataFrames
        self.df_merging = pd.DataFrame(merging_data, columns=['name', 'type', 'compilation', 'semantic', 'conditional', 'joint'])
        
        model_data = list(zip(models, compilation_rate, semantic_rate, conditioned_p, joint_p))
        self.df_models = pd.DataFrame(model_data, columns=['name', 'compilation', 'semantic', 'conditional', 'joint'])
        self.df_models['type'] = self.df_models['name'].apply(lambda x: 'arcee_fusion' if 'arcee' in x 
                                                             else 'model_stock' if 'stock' in x 
                                                             else 'della')
        
        # Combine datasets
        self.df_combined = pd.concat([self.df_merging, self.df_models], ignore_index=True)
        
    def create_scatter_plots(self, x_metric='compilation', y_metric='semantic'):
        """Create interactive scatter plots"""
        df = self.df_combined
            
        fig = px.scatter(df, x=x_metric, y=y_metric, color='type',
                        hover_name='name',
                        hover_data={'compilation': ':.3f', 'semantic': ':.3f', 
                                  'conditional': ':.3f', 'joint': ':.3f'},
                        color_discrete_map=self.colors,
                        title=f'{x_metric.title()} vs {y_metric.title()} - Combined Dataset',
                        labels={x_metric: f'{x_metric.title()} Rate', 
                               y_metric: f'{y_metric.title()} Rate'})
        
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='white')))
        fig.update_layout(height=600, width=800)
        return fig
    
    def create_heatmap(self):
        """Create heatmap visualization"""
        df = self.df_combined
            
        # Prepare data for heatmap
        metrics = ['compilation', 'semantic', 'conditional', 'joint']
        heatmap_data = df[metrics].values
        model_names = df['name'].values
        
        fig = px.imshow(heatmap_data, 
                       labels=dict(x="Metrics", y="Models", color="Performance"),
                       x=metrics,
                       y=model_names,
                       color_continuous_scale="Blues",
                       title=f"Performance Heatmap - Combined Dataset")
        
        fig.update_layout(height=max(400, len(model_names) * 25), width=800)
        return fig
    
    def create_parallel_coordinates(self):
        """Create parallel coordinates plot"""
        df = self.df_combined
            
        # Create color mapping
        unique_types = df['type'].unique()
        color_map = {t: i for i, t in enumerate(unique_types)}
        df['type_numeric'] = df['type'].map(color_map)
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df['type_numeric'],
                        # Change from 'Set1' to a valid colorscale
                        colorscale='viridis',  
                        showscale=True,
                        colorbar=dict(title="Model Type")),
                dimensions=list([
                    dict(range=[0, 1], label='Compilation', values=df['compilation']),
                    dict(range=[0, 1], label='Semantic', values=df['semantic']),
                    dict(range=[0, 1], label='Conditional', values=df['conditional']),
                    dict(range=[0, 1], label='Joint', values=df['joint'])
                ])
            )
        )
        
        fig.update_layout(
            title=f"Parallel Coordinates - Combined Dataset",
            height=600, width=800
        )
        
        return fig
    
    def create_performance_ranking(self, metric='compilation'):
        """Create ranking visualization"""
        df = self.df_combined.copy()
            
        df_sorted = df.sort_values(metric, ascending=True)
        
        fig = px.bar(df_sorted, x=metric, y='name', color='type',
                    orientation='h',
                    color_discrete_map=self.colors,
                    title=f"Model Ranking by {metric.title()} - Combined Dataset",
                    labels={metric: f'{metric.title()} Rate'})
        
        fig.update_layout(height=max(400, len(df) * 25), width=800)
        return fig

# Initialize dashboard
dashboard = ModelVisualizationDashboard()

# Streamlit app
def main():
    st.set_page_config(page_title="Model Performance Dashboard", page_icon="📊", layout="wide")
    
    st.title("📜 Model Performance Visualization Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    plot_type = st.sidebar.selectbox(
        'Plot Type:',
        ['Scatter Plot', 'Heatmap', 'Parallel Coordinates', 'Performance Ranking']
    )
    
    # Conditional controls based on plot type
    x_metric = 'compilation'
    y_metric = 'semantic'
    ranking_metric = 'compilation'
    
    if plot_type == 'Scatter Plot':
        x_metric = st.sidebar.selectbox(
            'X-axis:',
            ['compilation', 'semantic', 'conditional', 'joint'],
            index=0
        )
        y_metric = st.sidebar.selectbox(
            'Y-axis:',
            ['compilation', 'semantic', 'conditional', 'joint'],
            index=1
        )
    
    if plot_type == 'Performance Ranking':
        ranking_metric = st.sidebar.selectbox(
            'Rank by:',
            ['compilation', 'semantic', 'conditional', 'joint'],
            index=0
        )
    
    # Display the selected visualization
    if plot_type == 'Scatter Plot':
        st.plotly_chart(dashboard.create_scatter_plots(x_metric, y_metric), use_container_width=True)
    elif plot_type == 'Heatmap':
        st.plotly_chart(dashboard.create_heatmap(), use_container_width=True)
    elif plot_type == 'Parallel Coordinates':
        st.plotly_chart(dashboard.create_parallel_coordinates(), use_container_width=True)
    elif plot_type == 'Performance Ranking':
        st.plotly_chart(dashboard.create_performance_ranking(ranking_metric), use_container_width=True)
    
    # Add data details section
    with st.expander("Dataset Details"):
        st.dataframe(dashboard.df_combined)
    
    # Add summary report
    st.markdown("---")
    # st.header("📊 Performance Analysis Summary")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.subheader("🏆 Best Performers")
        
    #     # Best performers by compilation
    #     best_model = dashboard.df_combined.loc[dashboard.df_combined['compilation'].idxmax()]
        
    #     st.markdown(f"**Best Compilation Rate:**  \n"
    #                f"{best_model['name']} ({best_model['compilation']:.3f})")
        
    #     best_semantic = dashboard.df_combined.loc[dashboard.df_combined['semantic'].idxmax()]
    #     st.markdown(f"**Best Semantic Rate:**  \n"
    #                f"{best_semantic['name']} ({best_semantic['semantic']:.3f})")
    
    # with col2:
    #     st.subheader("🔍 Method Analysis")
        
    #     # Method analysis
    #     method_stats = dashboard.df_combined.groupby('type')[['compilation', 'semantic', 'conditional', 'joint']].mean()
        
    #     for method, stats in method_stats.iterrows():
    #         st.markdown(f"**{method.upper()}:**  \n"
    #                    f"Compilation={stats['compilation']:.3f}, Semantic={stats['semantic']:.3f}")
    
    # # Correlation insights
    # st.subheader("📈 Key Correlations")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     correlations = dashboard.df_combined[['compilation', 'semantic', 'conditional', 'joint']].corr()
    #     st.markdown(f"**Compilation vs Semantic:**  \n"
    #               f"Correlation: {correlations.loc['compilation', 'semantic']:.3f}")
    #     st.markdown(f"**Conditional vs Joint:**  \n"
    #               f"Correlation: {correlations.loc['conditional', 'joint']:.3f}")
    
    # with col2:
    #     # Create correlation heatmap
    #     fig = plt.figure(figsize=(6, 5))
    #     sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    #     plt.title('Correlation Matrix')
    #     st.pyplot(fig)
    
    # # Methodology explanations
    # st.markdown("---")
    # st.header("📚 Methodology Explanations")
    
    # methodologies = {
    #     "DARE": "Drop And REscale - probabilistically drops/rescales parameters",
    #     "TIES": "Task-wise Intersection Enhancement - resolves parameter conflicts",
    #     "Linear": "Simple parameter averaging (often fails)",
    #     "Magnitude Pruning": "Keeps parameters with highest magnitudes",
    #     "Della": "Delta-based merging with various configurations"
    # }
    
    # for method, description in methodologies.items():
    #     st.markdown(f"**{method}**: {description}")
    
    # # Footer
    # st.markdown("---")
    # st.markdown("### Usage Tips")
    # st.markdown("""
    # 1. Change 'Plot Type' to explore different visualizations
    # 2. For scatter plots, adjust X-axis and Y-axis metrics
    # 3. Use 'Rank by' dropdown for performance ranking plots
    # 4. Expand the Dataset Details section to view raw data
    # """)

if __name__ == '__main__':
    main()