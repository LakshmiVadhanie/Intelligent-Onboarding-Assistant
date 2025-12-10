"""
Monitoring Dashboard - Real-time visualization of model metrics
Built with Plotly and Streamlit
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.data_drift_detector import DataDriftDetector
from src.monitoring.alert_system import ThresholdAlertSystem


def load_components():
    """Load monitoring components"""
    metrics_collector = MetricsCollector()
    drift_detector = DataDriftDetector()
    alert_system = ThresholdAlertSystem()
    return metrics_collector, drift_detector, alert_system


def plot_metrics_over_time(metrics_collector: MetricsCollector,
                          metric_name: str,
                          limit: int = 100):
    """Plot a metric over time"""
    history = metrics_collector.get_metrics_history(limit=limit)
    
    if not history:
        st.info(f"No data available for {metric_name}")
        return
    
    # Extract metric and timestamps
    data = []
    for record in history:
        if metric_name in record:
            data.append({
                'timestamp': record['timestamp'],
                'value': record[metric_name]
            })
    
    if not data:
        st.info(f"No data available for {metric_name}")
        return
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['value'],
        mode='lines+markers',
        name=metric_name,
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))
    
    # Add threshold line if available
    alert_system = ThresholdAlertSystem()
    if metric_name in alert_system.thresholds:
        threshold = alert_system.thresholds[metric_name]
        fig.add_hline(
            y=threshold.lower_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Lower Threshold: {threshold.lower_threshold:.4f}",
            annotation_position="right"
        )
        
        if threshold.upper_threshold:
            fig.add_hline(
                y=threshold.upper_threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Upper Threshold: {threshold.upper_threshold:.4f}",
                annotation_position="right"
            )
    
    fig.update_layout(
        title=f"{metric_name} Over Time",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_metric_distribution(metrics_collector: MetricsCollector,
                             metric_name: str,
                             limit: int = 100):
    """Plot distribution of a metric"""
    stats = metrics_collector.compute_metric_statistics(metric_name, limit=limit)
    
    history = metrics_collector.get_metrics_history(limit=limit)
    values = [m.get(metric_name, 0) for m in history if metric_name in m]
    
    if not values:
        st.info(f"No data available for {metric_name}")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        name=metric_name,
        marker=dict(color='#667eea')
    ))
    
    fig.update_layout(
        title=f"{metric_name} Distribution",
        xaxis_title="Value",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{stats['mean']:.4f}")
    with col2:
        st.metric("Std Dev", f"{stats['std']:.4f}")
    with col3:
        st.metric("Min", f"{stats['min']:.4f}")
    with col4:
        st.metric("Max", f"{stats['max']:.4f}")


def plot_drift_timeline(drift_detector: DataDriftDetector):
    """Plot drift detection timeline"""
    drift_summary = drift_detector.get_drift_summary()
    history = drift_detector.drift_history
    
    if not history:
        st.info("No drift data available")
        return
    
    # Prepare data for visualization
    data = []
    for record in history:
        data.append({
            'timestamp': record.get('timestamp', ''),
            'feature': record.get('metric_name', ''),
            'is_drift': record.get('is_drift_detected', False),
            'p_value': record.get('p_value', 0),
            'drift_score': record.get('drift_score', 0)
        })
    
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        # Add drift events
        drift_events = df[df['is_drift'] == True]
        no_drift_events = df[df['is_drift'] == False]
        
        fig.add_trace(go.Scatter(
            x=no_drift_events['timestamp'],
            y=no_drift_events['feature'],
            mode='markers',
            name='No Drift',
            marker=dict(size=8, color='green', symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=drift_events['timestamp'],
            y=drift_events['feature'],
            mode='markers',
            name='Drift Detected',
            marker=dict(size=12, color='red', symbol='diamond')
        ))
        
        fig.update_layout(
            title="Data Drift Detection Timeline",
            xaxis_title="Time",
            yaxis_title="Feature",
            hovermode='closest',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display drift summary
    st.markdown("### Drift Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Checks", drift_summary['total_checks'])
    with col2:
        st.metric("Drifts Detected", drift_summary['drift_count'])
    with col3:
        st.metric("Drift Rate", f"{drift_summary['drift_rate']:.2%}")


def plot_alerts_timeline(alert_system: ThresholdAlertSystem):
    """Plot alerts timeline"""
    alert_summary = alert_system.get_alert_summary()
    
    if not alert_system.alerts_history:
        st.info("No alerts triggered")
        return
    
    history = alert_system.alerts_history
    
    # Prepare data
    data = []
    severity_colors = {'info': 'blue', 'warning': 'orange', 'critical': 'red'}
    
    for alert in history:
        data.append({
            'timestamp': alert.get('timestamp', ''),
            'metric': alert.get('metric_name', ''),
            'severity': alert.get('severity', ''),
            'value': alert.get('current_value', 0)
        })
    
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.scatter(
            df,
            x='timestamp',
            y='metric',
            color='severity',
            color_discrete_map=severity_colors,
            hover_data=['value'],
            title="Alert Timeline",
            labels={'timestamp': 'Time', 'metric': 'Metric', 'severity': 'Severity'}
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display alert summary
    st.markdown("### Alert Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Alerts", alert_summary['total_alerts'])
    with col2:
        st.metric("Active Thresholds", alert_summary['active_thresholds'])
    
    # Display by severity
    if alert_summary['by_severity']:
        st.markdown("#### Alerts by Severity")
        for severity, count in alert_summary['by_severity'].items():
            st.write(f"**{severity.upper()}**: {count}")


def display_model_metrics(metrics_collector: MetricsCollector):
    """Display current model metrics"""
    latest = metrics_collector.get_latest_metrics()
    
    if not latest:
        st.info("No metrics available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Precision@5",
            f"{latest.get('precision_at_5', 0):.4f}",
            delta=f"{(latest.get('precision_at_5', 0) - 0.5):.4f}" if 'precision_at_5' in latest else None
        )
    
    with col2:
        st.metric(
            "Recall@5",
            f"{latest.get('recall_at_5', 0):.4f}"
        )
    
    with col3:
        st.metric(
            "F1@5",
            f"{latest.get('f1_at_5', 0):.4f}"
        )
    
    with col4:
        st.metric(
            "MRR",
            f"{latest.get('mrr', 0):.4f}"
        )
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "NDCG@10",
            f"{latest.get('ndcg_at_10', 0):.4f}"
        )
    
    with col6:
        st.metric(
            "Response Time",
            f"{latest.get('response_time', 0):.2f}s"
        )
    
    with col7:
        st.metric(
            "Relevance Score",
            f"{latest.get('avg_relevance_score', 0):.4f}"
        )
    
    with col8:
        st.metric(
            "Query Count",
            f"{latest.get('query_count', 0)}"
        )


def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Model Monitoring Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        * {
            font-family: 'Inter', sans-serif;
        }
        .main {
            background-color: #0f1117;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Model Monitoring Dashboard")
    
    # Load components
    metrics_collector, drift_detector, alert_system = load_components()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Metrics", "Data Drift", "Alerts", "Thresholds", "Export Data"]
    )
    
    # Overview Page
    if page == "Overview":
        st.header("Model Performance Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Current Metrics")
            display_model_metrics(metrics_collector)
        
        with col2:
            st.subheader("Quick Stats")
            history = metrics_collector.get_metrics_history(limit=100)
            st.write(f"**Total Queries Monitored**: {len(history)}")
            st.write(f"**Total Alerts**: {alert_system.get_alert_summary()['total_alerts']}")
            st.write(f"**Drift Detections**: {drift_detector.get_drift_summary()['drift_count']}")
    
    # Metrics Page
    elif page == "Metrics":
        st.header("Performance Metrics Analysis")
        
        metrics_list = ['precision_at_5', 'precision_at_10', 'recall_at_5', 'recall_at_10',
                       'f1_at_5', 'f1_at_10', 'mrr', 'ndcg_at_10', 'avg_response_time',
                       'avg_relevance_score']
        
        selected_metric = st.selectbox("Select Metric", metrics_list)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time Series")
            plot_metrics_over_time(metrics_collector, selected_metric)
        
        with col2:
            st.subheader("Distribution")
            plot_metric_distribution(metrics_collector, selected_metric)
    
    # Data Drift Page
    elif page == "Data Drift":
        st.header("Data Drift Monitoring")
        plot_drift_timeline(drift_detector)
        
        st.subheader("Recent Drift Reports")
        history = drift_detector.get_drift_summary(limit=10)
        if history['recent_drifts']:
            df = pd.DataFrame(history['recent_drifts'])
            st.dataframe(df, use_container_width=True)
    
    # Alerts Page
    elif page == "Alerts":
        st.header("Alert Monitoring")
        plot_alerts_timeline(alert_system)
        
        st.subheader("Recent Alerts")
        summary = alert_system.get_alert_summary(limit=20)
        if summary['recent_alerts']:
            df = pd.DataFrame(summary['recent_alerts'])
            st.dataframe(df, use_container_width=True)
    
    # Thresholds Page
    elif page == "Thresholds":
        st.header("Alert Thresholds Configuration")
        
        for metric_name, threshold in alert_system.thresholds.items():
            with st.expander(f"{metric_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Lower Threshold**: {threshold.lower_threshold:.4f}")
                
                with col2:
                    if threshold.upper_threshold:
                        st.write(f"**Upper Threshold**: {threshold.upper_threshold:.4f}")
                    else:
                        st.write("**Upper Threshold**: None")
                
                with col3:
                    st.write(f"**Severity**: {threshold.severity}")
                
                st.write(f"**Enabled**: {threshold.enabled}")
    
    # Export Data Page
    elif page == "Export Data":
        st.header("Export Monitoring Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics Export")
            export_format = st.selectbox("Metrics Format", ["json", "csv"], key="metrics_format")
            if st.button("Export Metrics"):
                export_path = f"monitoring_export/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                metrics_collector.export_metrics(export_path, format=export_format)
                st.success(f"Metrics exported to {export_path}")
        
        with col2:
            st.subheader("Alerts Export")
            export_format = st.selectbox("Alerts Format", ["json", "csv"], key="alerts_format")
            if st.button("Export Alerts"):
                export_path = f"monitoring_export/alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                alert_system.export_alerts(export_path, format=export_format)
                st.success(f"Alerts exported to {export_path}")


if __name__ == "__main__":
    main()
