import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Overtake Analysis",
    layout="wide"
)

st.title("Overtake Chance Analysis")
st.markdown("*Based on car size ratio + approach detection*")

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('overtake_data.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("No data found! Please run the tracker first.")
    st.info("Run: `python overtake_tracker.py` to generate data")
    st.stop()

# ============================================
# SIDEBAR - QUICK METRICS
# ============================================

with st.sidebar:
    st.header("Quick Stats")
    
    total_frames = len(df)
    
    if 'overtake_chance' in df.columns:
        max_chance = df['overtake_chance'].max()
        avg_chance = df['overtake_chance'].mean()
        st.metric("Max Overtake Chance", f"{max_chance:.0f}%")
        st.metric("Avg Overtake Chance", f"{avg_chance:.1f}%")
    
    if 'area_ratio' in df.columns:
        max_ratio = df['area_ratio'].max()
        st.metric("Max Size Ratio", f"{max_ratio:.2f}x")
    
    st.metric("Total Frames", f"{total_frames:,}")
    
    # Best overtake moment
    if 'overtake_chance' in df.columns and 'frame' in df.columns:
        best_row = df.loc[df['overtake_chance'].idxmax()]
        st.divider()
        st.metric("Best Overtake", f"{best_row['overtake_chance']:.0f}%")
        if 'area_ratio' in df.columns:
            st.caption(f"Frame {best_row['frame']:.0f} | Size: {best_row['area_ratio']:.2f}x")

# ============================================
# MAIN TABS
# ============================================

tab1, tab2, tab3 = st.tabs(["Overview", "Trends", "Stats"])

# ============================================
# TAB 1: OVERVIEW
# ============================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Overtake Chance Over Time")
        
        if 'overtake_chance' in df.columns and 'frame' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df['overtake_chance'],
                mode='lines',
                name='Overtake Chance',
                fill='tozeroy',
                line=dict(color='#00cc96', width=2)
            ))
            
            # Add threshold lines
            fig.add_hline(y=70, line_dash="dash", line_color="green", 
                          annotation_text="High (70%)")
            fig.add_hline(y=45, line_dash="dash", line_color="yellow", 
                          annotation_text="Medium (45%)")
            fig.add_hline(y=25, line_dash="dash", line_color="orange", 
                          annotation_text="Low (25%)")
            
            fig.update_layout(
                height=400,
                xaxis_title="Frame",
                yaxis_title="Overtake Chance (%)",
                yaxis_range=[0, 100],
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No overtake_chance data found")
    
    with col2:
        st.subheader("Status Breakdown")
        
        # Check which status column exists
        status_col = None
        if 'status' in df.columns:
            status_col = 'status'
        elif 'overtake_status' in df.columns:
            status_col = 'overtake_status'
        
        if status_col:
            status_counts = df[status_col].value_counts()
            
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No status column found")

# ============================================
# TAB 2: TRENDS
# ============================================

with tab2:
    st.subheader("Detailed Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'area_ratio' in df.columns and 'frame' in df.columns:
            st.subheader("Car Size Ratio")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df['area_ratio'],
                mode='lines',
                name='Size Ratio',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Threshold lines
            fig.add_hline(y=1.5, line_dash="dash", line_color="green", 
                          annotation_text="Overtake Starts (1.5x)")
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                          annotation_text="Initial Size (1.0x)")
            fig.add_hline(y=3.0, line_dash="dash", line_color="red", 
                          annotation_text="Max Chance (3.0x)")
            
            fig.update_layout(
                height=300,
                xaxis_title="Frame",
                yaxis_title="Size Ratio (x)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'status' in df.columns and 'frame' in df.columns:
            st.subheader("Approach Status")
            
            # Create a numeric column for the timeline (all 1s)
            df['dummy_y'] = 1
            
            # Create color map
            unique_statuses = df['status'].unique()
            colors = ['green' if 'APPROACH' in str(s) else 'red' if 'MOVING' in str(s) else 'gray' for s in unique_statuses]
            color_map = dict(zip(unique_statuses, colors))
            
            fig = px.scatter(
                df,
                x='frame',
                y='dummy_y',
                color='status',
                color_discrete_map=color_map,
                title="Approach Status Timeline",
                labels={'frame': 'Frame', 'dummy_y': ''}
            )
            fig.update_layout(
                height=150,
                showlegend=True,
                yaxis_visible=False,
                yaxis_showticklabels=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Clean up
            df = df.drop(columns=['dummy_y'])
    
    # Duration info
    if 'duration_penalty' in df.columns and 'frame' in df.columns:
        st.subheader("Duration Penalty")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df['duration_penalty'],
            mode='lines',
            name='Duration Penalty',
            fill='tozeroy',
            line=dict(color='red', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=10, line_dash="dash", line_color="yellow", 
                      annotation_text="Penalty Starts")
        
        fig.update_layout(
            height=200,
            xaxis_title="Frame",
            yaxis_title="Duration Penalty (%)",
            yaxis_range=[0, 25]
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: STATISTICS
# ============================================

with tab3:
    st.subheader("Statistics")
    
    if 'overtake_chance' in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{df['overtake_chance'].mean():.1f}%")
        with col2:
            st.metric("Median", f"{df['overtake_chance'].median():.1f}%")
        with col3:
            st.metric("Max", f"{df['overtake_chance'].max():.0f}%")
        with col4:
            st.metric("Min", f"{df['overtake_chance'].min():.0f}%")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of overtake chances
            st.subheader("Chance Distribution")
            fig = px.histogram(
                df,
                x='overtake_chance',
                nbins=20,
                title="Distribution of Overtake Chances",
                labels={'overtake_chance': 'Overtake Chance (%)', 'count': 'Frequency'},
                color_discrete_sequence=['#00cc96']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by status
            if 'status' in df.columns:
                st.subheader("Chance by Status")
                fig = px.box(
                    df,
                    x='status',
                    y='overtake_chance',
                    title="Overtake Chance by Approach Status",
                    color='status'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Size vs Chance scatter
        if 'area_ratio' in df.columns:
            st.subheader("Size vs Overtake Chance")
            fig = px.scatter(
                df,
                x='area_ratio',
                y='overtake_chance',
                color='overtake_chance',
                title="Overtake Chance vs Car Size",
                labels={'area_ratio': 'Size Ratio (x)', 'overtake_chance': 'Overtake Chance (%)'},
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No overtake_chance data found. Showing available columns:")
        st.write(df.head())

# ============================================
# RAW DATA (Expandable)
# ============================================

with st.expander("Raw Data"):
    st.dataframe(df)

# ============================================
# DOWNLOAD
# ============================================

st.divider()

if st.button("Download Data as CSV"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="overtake_analysis.csv",
        mime="text/csv"
    )

st.caption("The tracker compares 5-frame blocks. If the current block is bigger than the previous block, the car is APPROACHING.")