import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from math import sqrt

# Set page configuration
st.set_page_config(
    page_title="A/B Test Statistical Significance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 A/B Test Statistical Significance Analyzer")
st.markdown("""
This tool helps you analyze the statistical significance of your A/B test results.
Input your test data below and get comprehensive statistical analysis with actionable insights.
""")

# Sidebar for configuration
st.sidebar.header("Test Configuration")
significance_level = st.sidebar.selectbox(
    "Significance Level (α)",
    options=[0.05, 0.01, 0.001],
    index=0,
    help="The threshold for statistical significance. Lower values require stronger evidence."
)

test_type = st.sidebar.selectbox(
    "Test Type",
    options=["Two-tailed", "One-tailed (A > B)", "One-tailed (B > A)"],
    index=0,
    help="Choose the type of hypothesis test based on your research question."
)

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    options=["Conversion Rates", "Raw Counts"],
    help="Choose how you want to input your data"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Group A (Control)")
    if input_method == "Conversion Rates":
        sample_size_a = st.number_input(
            "Sample Size A",
            min_value=1,
            value=1000,
            help="Total number of visitors/users in Group A"
        )
        conversion_rate_a = st.number_input(
            "Conversion Rate A (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Percentage of users who converted in Group A"
        ) / 100
        conversions_a = int(sample_size_a * conversion_rate_a)
    else:
        sample_size_a = st.number_input(
            "Sample Size A",
            min_value=1,
            value=1000,
            help="Total number of visitors/users in Group A"
        )
        conversions_a = st.number_input(
            "Conversions A",
            min_value=0,
            max_value=sample_size_a,
            value=50,
            help="Number of users who converted in Group A"
        )
        conversion_rate_a = conversions_a / sample_size_a if sample_size_a > 0 else 0

with col2:
    st.subheader("📈 Group B (Treatment)")
    if input_method == "Conversion Rates":
        sample_size_b = st.number_input(
            "Sample Size B",
            min_value=1,
            value=1000,
            help="Total number of visitors/users in Group B"
        )
        conversion_rate_b = st.number_input(
            "Conversion Rate B (%)",
            min_value=0.0,
            max_value=100.0,
            value=6.0,
            step=0.1,
            help="Percentage of users who converted in Group B"
        ) / 100
        conversions_b = int(sample_size_b * conversion_rate_b)
    else:
        sample_size_b = st.number_input(
            "Sample Size B",
            min_value=1,
            value=1000,
            help="Total number of visitors/users in Group B"
        )
        conversions_b = st.number_input(
            "Conversions B",
            min_value=0,
            max_value=sample_size_b,
            value=60,
            help="Number of users who converted in Group B"
        )
        conversion_rate_b = conversions_b / sample_size_b if sample_size_b > 0 else 0

# Validation
if sample_size_a <= 0 or sample_size_b <= 0:
    st.error("Sample sizes must be greater than 0")
    st.stop()

if conversions_a > sample_size_a or conversions_b > sample_size_b:
    st.error("Conversions cannot exceed sample size")
    st.stop()

# Calculate statistics
def calculate_ab_test_stats(n_a, x_a, n_b, x_b, alpha=0.05, test_type="two-tailed"):
    """
    Calculate A/B test statistics using chi-square test and z-test for proportions
    """
    # Conversion rates
    p_a = x_a / n_a
    p_b = x_b / n_b
    
    # Pooled proportion for z-test
    p_pooled = (x_a + x_b) / (n_a + n_b)
    
    # Standard error
    se = sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    
    # Z-score
    z_score = (p_b - p_a) / se if se > 0 else 0
    
    # Chi-square test
    observed = np.array([[x_a, n_a - x_a], [x_b, n_b - x_b]])
    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(observed)
    
    # Z-test p-value
    if test_type == "two-tailed":
        p_value_z = 2 * (1 - stats.norm.cdf(abs(z_score)))
    elif test_type == "one-tailed (A > B)":
        p_value_z = stats.norm.cdf(z_score)
    else:  # one-tailed (B > A)
        p_value_z = 1 - stats.norm.cdf(z_score)
    
    # Effect size (relative change)
    relative_change = (p_b - p_a) / p_a * 100 if p_a > 0 else 0
    
    # Confidence interval for difference in proportions
    diff = p_b - p_a
    se_diff = sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    
    if test_type == "two-tailed":
        z_critical = stats.norm.ppf(1 - alpha/2)
    else:
        z_critical = stats.norm.ppf(1 - alpha)
    
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff
    
    # Statistical power calculation (post-hoc)
    effect_size = abs(p_b - p_a) / sqrt(p_pooled * (1 - p_pooled))
    power = stats.norm.cdf(z_critical - effect_size * sqrt(n_a * n_b / (n_a + n_b)))
    
    return {
        'p_a': p_a,
        'p_b': p_b,
        'p_pooled': p_pooled,
        'z_score': z_score,
        'p_value_z': p_value_z,
        'p_value_chi2': p_value_chi2,
        'relative_change': relative_change,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'power': power,
        'effect_size': effect_size,
        'chi2': chi2
    }

# Calculate results
try:
    results = calculate_ab_test_stats(
        sample_size_a, conversions_a, sample_size_b, conversions_b,
        significance_level, test_type.lower()
    )
    
    # Determine significance
    is_significant = results['p_value_z'] < significance_level
    
    # Results display
    st.markdown("---")
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Group A Conversion Rate",
            f"{results['p_a']:.2%}",
            f"{conversions_a}/{sample_size_a}"
        )
    
    with col2:
        st.metric(
            "Group B Conversion Rate",
            f"{results['p_b']:.2%}",
            f"{conversions_b}/{sample_size_b}"
        )
    
    with col3:
        delta_color = "normal" if not is_significant else ("inverse" if results['relative_change'] < 0 else "normal")
        st.metric(
            "Relative Change",
            f"{results['relative_change']:+.1f}%",
            delta_color=delta_color
        )
    
    with col4:
        st.metric(
            "Sample Size Total",
            f"{sample_size_a + sample_size_b:,}",
            f"A: {sample_size_a:,}, B: {sample_size_b:,}"
        )
    
    # Statistical significance result
    st.markdown("### 🎯 Statistical Significance")
    
    if is_significant:
        st.success(f"✅ **STATISTICALLY SIGNIFICANT** at α = {significance_level}")
        st.markdown(f"""
        **P-value: {results['p_value_z']:.4f}** (< {significance_level})
        
        The difference between Group A and Group B is statistically significant.
        You can be confident that the observed difference is not due to random chance.
        """)
    else:
        st.error(f"❌ **NOT STATISTICALLY SIGNIFICANT** at α = {significance_level}")
        st.markdown(f"""
        **P-value: {results['p_value_z']:.4f}** (≥ {significance_level})
        
        The difference between Group A and Group B is not statistically significant.
        The observed difference could be due to random chance.
        """)
    
    # Detailed statistics
    st.markdown("### 📈 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Test Statistics:**")
        st.write(f"• Z-score: {results['z_score']:.4f}")
        st.write(f"• Chi-square: {results['chi2']:.4f}")
        st.write(f"• P-value (Z-test): {results['p_value_z']:.4f}")
        st.write(f"• P-value (Chi-square): {results['p_value_chi2']:.4f}")
        st.write(f"• Effect size: {results['effect_size']:.4f}")
    
    with col2:
        st.markdown("**Confidence Interval:**")
        ci_percentage = (1 - significance_level) * 100
        st.write(f"• {ci_percentage:.0f}% CI for difference: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        st.write(f"• {ci_percentage:.0f}% CI for difference (%): [{results['ci_lower']*100:.2f}%, {results['ci_upper']*100:.2f}%]")
        st.write(f"• Statistical power: {results['power']:.2%}")
        st.write(f"• Pooled conversion rate: {results['p_pooled']:.4f}")
    
    # Visualization
    st.markdown("### 📊 Visualization")
    
    # Create comparison chart
    fig = go.Figure()
    
    groups = ['Group A (Control)', 'Group B (Treatment)']
    rates = [results['p_a'], results['p_b']]
    colors = ['#ff6b6b', '#4ecdc4']
    
    fig.add_trace(go.Bar(
        x=groups,
        y=rates,
        marker_color=colors,
        text=[f"{rate:.2%}" for rate in rates],
        textposition='auto',
        name='Conversion Rate'
    ))
    
    fig.update_layout(
        title='Conversion Rate Comparison',
        yaxis_title='Conversion Rate',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence interval visualization
    fig_ci = go.Figure()
    
    fig_ci.add_trace(go.Scatter(
        x=[results['ci_lower'], results['ci_upper']],
        y=[0, 0],
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=4),
        name='95% Confidence Interval'
    ))
    
    fig_ci.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="No Effect", annotation_position="top")
    
    fig_ci.update_layout(
        title=f'{(1-significance_level)*100:.0f}% Confidence Interval for Difference in Conversion Rates',
        xaxis_title='Difference in Conversion Rate',
        yaxis_title='',
        showlegend=False,
        height=200,
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig_ci, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if is_significant:
        if results['relative_change'] > 0:
            st.success("""
            **Recommendation: Implement Group B (Treatment)**
            
            • Group B shows a statistically significant improvement
            • The positive effect is unlikely to be due to chance
            • Consider rolling out the treatment to all users
            • Monitor long-term effects and key metrics
            """)
        else:
            st.error("""
            **Recommendation: Stick with Group A (Control)**
            
            • Group B shows a statistically significant decrease
            • The treatment appears to have a negative effect
            • Do not implement the treatment
            • Consider investigating why the treatment performed poorly
            """)
    else:
        st.warning("""
        **Recommendation: Continue Testing or Investigate**
        
        • The test is inconclusive - no significant difference detected
        • Consider running the test longer to increase sample size
        • Evaluate if the effect size is practically meaningful
        • You may need a larger sample size to detect smaller effects
        """)
    
    # Sample size recommendations
    st.markdown("### 📏 Sample Size Analysis")
    
    # Calculate required sample size for 80% power
    def calculate_sample_size_per_group(p1, p2, alpha=0.05, power=0.8):
        """Calculate required sample size per group for given power"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p_avg = (p1 + p2) / 2
        effect_size = abs(p2 - p1)
        
        if effect_size == 0:
            return float('inf')
        
        n = (z_alpha * sqrt(2 * p_avg * (1 - p_avg)) + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / effect_size**2
        return int(np.ceil(n))
    
    required_n = calculate_sample_size_per_group(results['p_a'], results['p_b'])
    
    if required_n != float('inf'):
        current_n = min(sample_size_a, sample_size_b)
        st.write(f"• Required sample size per group (80% power): {required_n:,}")
        st.write(f"• Current sample size per group: {current_n:,}")
        if current_n < required_n:
            st.write(f"• **Recommended:** Increase sample size by {required_n - current_n:,} per group")
    
    # Export results
    st.markdown("### 📋 Export Results")
    
    results_df = pd.DataFrame({
        'Metric': [
            'Group A Sample Size', 'Group A Conversions', 'Group A Conversion Rate',
            'Group B Sample Size', 'Group B Conversions', 'Group B Conversion Rate',
            'Relative Change (%)', 'Z-Score', 'P-Value', 'Statistically Significant',
            'Confidence Interval Lower', 'Confidence Interval Upper', 'Statistical Power'
        ],
        'Value': [
            sample_size_a, conversions_a, f"{results['p_a']:.4f}",
            sample_size_b, conversions_b, f"{results['p_b']:.4f}",
            f"{results['relative_change']:.2f}%", f"{results['z_score']:.4f}",
            f"{results['p_value_z']:.4f}", "Yes" if is_significant else "No",
            f"{results['ci_lower']:.4f}", f"{results['ci_upper']:.4f}",
            f"{results['power']:.2%}"
        ]
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    # Convert to CSV for download
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="ab_test_results.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error calculating statistics: {str(e)}")
    st.write("Please check your inputs and try again.")

# Footer with explanation
st.markdown("---")
