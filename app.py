import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ==========================================
# 0. KONFIGURASI & CUSTOM CSS (Satu Data UHO Clone)
# ==========================================
st.set_page_config(page_title="UHO Analytics", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Import Fonts from Satu Data UHO Reference */
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
        background-color: #0F172A; /* Midnight Slate background */
        color: #E2E8F0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Instrument Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #F8FAFC;
    }
    
    /* Header Typography */
    .sd-title {
        font-family: 'Instrument Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
        letter-spacing: -0.5px;
    }
    .sd-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 30px;
    }
    
    /* Tailwind-like Flat Cards WITH ANIMATION */
    .sd-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .sd-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
        border-color: #38BDF8;
    }
    /* Add a subtle glow on hover to the cards */
    .sd-card::after {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.4s ease;
    }
    .sd-card:hover::after {
        transform: scaleX(1);
    }
    
    /* Native Streamlit Bordered Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1E293B;
        border: 1px solid #334155 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 20px -5px rgba(0, 0, 0, 0.4);
        border-color: #38BDF8 !important;
    }
    
    /* Form Container */
    [data-testid="stForm"] {
        background-color: #1E293B;
        border: 1px solid #334155 !important;
        border-radius: 16px !important;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .sd-kpi-title {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .sd-kpi-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Instrument Sans', sans-serif;
        color: #F8FAFC;
    }
    
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-bottom: 20px;
    }
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .sd-card { margin-bottom: 0px; }
    }
    @media (max-width: 480px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        .sd-kpi-title { font-size: 0.65rem; }
        .sd-kpi-value { font-size: 1.5rem; }
        .sd-card { padding: 16px; margin-bottom: 0px; }
    }
    
    
    /* Minimalist Profile Detail Card */
    .profile-header {
        border-bottom: 1px solid #334155;
        padding-bottom: 16px;
        margin-bottom: 16px;
    }
    .profile-name {
        font-family: 'Instrument Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .profile-meta {
        font-size: 0.95rem;
        color: #38BDF8;
        font-weight: 600;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
    }
    .metric-box {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        text-align: left;
        transition: all 0.2s ease-in-out;
    }
    .metric-box:hover {
        background-color: #172554;
        border-color: #1E40AF;
        transform: translateY(-3px);
        box-shadow: 0 4px 6px -1px rgba(56, 189, 248, 0.1);
    }
    .m-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        transition: color 0.2s ease;
    }
    .metric-box:hover .m-label {
        color: #38BDF8;
    }
    .m-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'Instrument Sans', sans-serif;
    }
    
    /* Badges */
    .sd-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .badge-excellent { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #059669; }
    .badge-good { background-color: rgba(56, 189, 248, 0.2); color: #7DD3FC; border: 1px solid #0284C7; }
    .badge-fair { background-color: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid #D97706; }
    .badge-poor { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid #DC2626; }
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CACHING & MODEL TRAINING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('dosen_uho_ml_ready.csv')
    return df

@st.cache_resource
def train_model(df):
    features = [
        'Prodi', 'Fakultas',
        'Sinta_Score', 'Sinta_Score_3Yr', 'Affil_Score', 'Affil_Score_3Yr', 
        'Total_Publikasi_GS', 'Sitasi_GS', 'h_index_GS', 'i10_index_GS',
        'Pub_2020', 'Pub_2021', 'Pub_2022', 'Pub_2023', 'Pub_2024', 'Pub_2025', 'Pub_2026'
    ]
    df_model = df.copy()
    
    df_model['Prodi'] = df_model['Prodi'].fillna(df_model['Prodi'].mode()[0])
    df_model['Fakultas'] = df_model['Fakultas'].fillna(df_model['Fakultas'].mode()[0])
    
    prodi_encoder = LabelEncoder()
    fakultas_encoder = LabelEncoder()
    
    df_model['Prodi'] = prodi_encoder.fit_transform(df_model['Prodi'].astype(str))
    df_model['Fakultas'] = fakultas_encoder.fit_transform(df_model['Fakultas'].astype(str))
    
    numeric_cols = features[2:]
    df_model[numeric_cols] = df_model[numeric_cols].fillna(0)
    
    X = df_model[features]
    y = df_model['Kategori_Kinerja']
    
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y_encoded)
    
    feat_imp = pd.DataFrame({'Fitur': features, 'Kepentingan': rf.feature_importances_}).sort_values('Kepentingan', ascending=True)
    
    return rf, y_encoder, prodi_encoder, fakultas_encoder, features, feat_imp

df = load_data()
rf_model, label_encoder, prodi_enc, fakultas_enc, feature_cols, feat_importance = train_model(df)

def predict_future_pubs(hist_data):
    X = np.array([[2020], [2021], [2022], [2023], [2024], [2025], [2026]])
    y = np.array(hist_data)
    model = LinearRegression()
    model.fit(X, y)
    
    X_future = np.array([[2027], [2028]])
    y_future = model.predict(X_future)
    return [max(0, int(round(val))) for val in y_future]

# ==========================================
# 2. HEADER & KPI (SATU DATA STYLE)
# ==========================================
st.markdown('<p class="sd-title">UHO Research & Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sd-subtitle">Pusat Intelijen Data & Prediksi Kinerja Tri Dharma Dosen Universitas Halu Oleo (Powered by ML)</p>', unsafe_allow_html=True)

# KPI Cards
total_dosen = len(df)
avg_sinta_3yr = int(df['Sinta_Score_3Yr'].mean())
total_pub_gs = int(df['Total_Publikasi_GS'].sum())
total_cit_gs = int(df['Sitasi_GS'].sum())

kpi_html = f"""
<div class="kpi-grid">
    <div class="sd-card"><div class="sd-kpi-title">Total Dosen Tersertifikasi</div><div class="sd-kpi-value">{f"{total_dosen:,}".replace(",", ".")}</div></div>
    <div class="sd-card"><div class="sd-kpi-title">Rata-rata Sinta Score 3 Tahun</div><div class="sd-kpi-value">{f"{avg_sinta_3yr:,}".replace(",", ".")}</div></div>
    <div class="sd-card"><div class="sd-kpi-title">Total Publikasi Universitas</div><div class="sd-kpi-value">{f"{total_pub_gs:,}".replace(",", ".")}</div></div>
    <div class="sd-card"><div class="sd-kpi-title">Total Sitasi Global</div><div class="sd-kpi-value">{f"{total_cit_gs:,}".replace(",", ".")}</div></div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ==========================================
# 3. TABS NAVIGATION
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Analitik Universitas", "🤖 Machine Learning Predictor", "🔎 Pencarian Dosen"])

# --- TAB 1: ANALITIK UNIVERSITAS ---
with tab1:
    st.markdown("### Performa Program Studi & Tren Akademik")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.container(border=True):
            st.markdown("**Komposisi Kelas Kinerja**")
            kelas_counts = df['Kategori_Kinerja'].value_counts().reset_index()
            kelas_counts.columns = ['Kategori', 'Jumlah']
            
            # ECharts Donut Chart with Hover Animation (Native HTML/JS Bypass)
            import json
            import streamlit.components.v1 as components
            
            color_map = {'Berpengaruh': '#10B981', 'Produktif': '#3B82F6', 'Cukup': '#F59E0B', 'Rendah': '#EF4444', 'Buruk': '#94A3B8'}
            echarts_data = [{"value": int(row['Jumlah']), "name": row['Kategori'], "itemStyle": {"color": color_map[row['Kategori']]}} for _, row in kelas_counts.iterrows()]
            data_json = json.dumps(echarts_data)
            
            echarts_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');
                    body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; font-family: 'Instrument Sans', sans-serif; }}
                </style>
            </head>
            <body>
                <div id="main" style="width: 100%; height: 350px;"></div>
                <script>
                    var chartDom = document.getElementById('main');
                    var myChart = echarts.init(chartDom);
                    var option = {{
                        tooltip: {{
                            trigger: 'item',
                            confine: true,
                            formatter: '{{b}}: {{c}} Dosen ({{d}}%)',
                            backgroundColor: '#0F172A',
                            textStyle: {{ color: '#E2E8F0', fontFamily: 'Instrument Sans', fontSize: 14 }},
                            borderWidth: 1,
                            borderColor: '#334155',
                            shadowBlur: 10,
                            shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }},
                        legend: {{
                            top: 'bottom',
                            textStyle: {{ fontFamily: 'Open Sans', color: '#94A3B8' }}
                        }},
                        series: [
                            {{
                                name: 'Kinerja',
                                type: 'pie',
                                radius: ['50%', '70%'],
                                center: ['50%', '42%'],
                                avoidLabelOverlap: false,
                                itemStyle: {{
                                    borderRadius: 5,
                                    borderColor: '#1E293B',
                                    borderWidth: 3
                                }},
                                label: {{ show: false, position: 'center' }},
                                emphasis: {{
                                    scaleSize: 10,
                                    label: {{
                                        show: true,
                                        fontSize: 16,
                                        fontWeight: 'bold',
                                        fontFamily: 'Instrument Sans',
                                        color: '#F8FAFC'
                                    }},
                                    itemStyle: {{
                                        shadowBlur: 15,
                                        shadowOffsetX: 0,
                                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                                    }}
                                }},
                                labelLine: {{ show: false }},
                                data: {data_json}
                            }}
                        ]
                    }};
                    myChart.setOption(option);
                    window.onresize = function() {{ myChart.resize(); }};
                </script>
            </body>
            </html>
            """
            components.html(echarts_html, height=360)
        
    with col2:
        with st.container(border=True):
            col_sel1, col_sel2, col_sel3 = st.columns([1.5, 1.5, 1])
            with col_sel1:
                metric_choice = st.selectbox("Indikator Perbandingan:", 
                                             ['Sinta_Score_3Yr', 'Sinta_Score', 'Total_Publikasi_GS', 'Sitasi_GS', 'h_index_GS'],
                                             format_func=lambda x: x.replace('_', ' ').replace('GS', 'Google Scholar').upper())
            with col_sel2:
                fakultas_filter = st.selectbox("Filter Fakultas:", ['Semua Fakultas'] + sorted(df['Fakultas'].dropna().unique().tolist()))
            with col_sel3:
                top_n = st.slider("Top N:", min_value=5, max_value=30, value=10, step=5)
                                             
            filtered_df = df if fakultas_filter == 'Semua Fakultas' else df[df['Fakultas'] == fakultas_filter]
            prodi_avg = filtered_df.groupby('Prodi')[metric_choice].mean().reset_index()
            top_prodi = prodi_avg.sort_values(by=metric_choice, ascending=False).head(top_n)
            
            # Bar Chart Bergradasi
            fig2 = px.bar(top_prodi, x=metric_choice, y='Prodi', orientation='h', 
                          color=metric_choice, color_continuous_scale=['#38BDF8', '#818CF8', '#4F46E5'], text=metric_choice)
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition="outside", 
                               marker_line_width=0, cliponaxis=False)
            chart_height = max(400, top_n * 35)
            fig2.update_layout(height=chart_height, yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10),
                               coloraxis_showscale=False, yaxis_title=None, xaxis_title=None,
                               xaxis=dict(showgrid=True, gridcolor='#334155', gridwidth=1, griddash='dash'),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#E2E8F0'))
            st.plotly_chart(fig2, use_container_width=True)

    # Tren Tahunan Area Chart
    with st.container(border=True):
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.markdown("**Tren Publikasi Universitas (2020 - 2026)**")
        with col_t2:
            prodi_trend_sel = st.multiselect("Bandingkan Pertumbuhan Prodi:", sorted(df['Prodi'].dropna().unique().tolist()), placeholder="Kosongkan untuk melihat total Universitas")
            
        years = ['Pub_2020', 'Pub_2021', 'Pub_2022', 'Pub_2023', 'Pub_2024', 'Pub_2025', 'Pub_2026']
        
        if not prodi_trend_sel:
            yearly_pubs = df[years].sum().reset_index()
            yearly_pubs.columns = ['Tahun', 'Jumlah Publikasi']
            yearly_pubs['Tahun'] = yearly_pubs['Tahun'].str.replace('Pub_', '')
            
            fig3 = px.area(yearly_pubs, x='Tahun', y='Jumlah Publikasi')
            fig3.update_traces(mode='lines+markers', line=dict(color='#10B981', width=4, shape='spline'),
                               marker=dict(size=8, color='#0F172A', line=dict(width=2, color='#10B981')),
                               fillcolor='rgba(16, 185, 129, 0.15)')
            fig3.add_annotation(x='2026', y=yearly_pubs['Jumlah Publikasi'].iloc[-1],
                                text="2026<br>(In Progress)", showarrow=True, arrowhead=1, ax=0, ay=-40,
                                font=dict(color="#EF4444", size=10), bgcolor="#1E293B", bordercolor="#EF4444", borderpad=4)
        else:
            trend_data = []
            for p in prodi_trend_sel:
                p_data = df[df['Prodi'] == p][years].sum().reset_index()
                p_data.columns = ['Tahun', 'Jumlah Publikasi']
                p_data['Tahun'] = p_data['Tahun'].str.replace('Pub_', '')
                p_data['Prodi'] = p
                trend_data.append(p_data)
            
            trend_df = pd.concat(trend_data)
            fig3 = px.line(trend_df, x='Tahun', y='Jumlah Publikasi', color='Prodi',
                           color_discrete_sequence=['#38BDF8', '#818CF8', '#34D399', '#F472B6', '#FBBF24'])
            fig3.update_traces(mode='lines+markers', line=dict(width=3, shape='spline'), marker=dict(size=6))
            
        fig3.update_layout(height=450, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           hovermode="x unified", xaxis_title=None, yaxis_title=None,
                           font=dict(color='#E2E8F0'),
                           xaxis=dict(showgrid=True, gridcolor='#334155', griddash='dash'),
                           yaxis=dict(showgrid=True, gridcolor='#334155', griddash='dash'),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""))
                           
        st.plotly_chart(fig3, use_container_width=True)

    # --- AUDIT & ANOMALY RADAR ---
    st.markdown("### 🛡️ Risk & Audit Intelligence")
    
    with st.container(border=True):
        st.markdown("<p style='font-size:0.95rem; color:#94A3B8; margin-bottom:15px;'>Radar pengawasan untuk mendeteksi Program Studi dengan tingkat profil dosen ghoib/tidak sinkron tertinggi.</p>", unsafe_allow_html=True)
        
        c_r1, c_r2 = st.columns([1, 1])
        risk_threshold = c_r1.slider("Ambang Batas Risiko (%)", min_value=0, max_value=100, value=0, step=5)
        search_prodi = c_r2.text_input("🔍 Cari Program Studi spesifik...", "")
        
        prodi_total = df.groupby('Prodi').size().reset_index(name='Total_Dosen')
        df['Is_Anomaly'] = df['Anomali'].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() == 'tidak' else 1)
        prodi_anomali = df.groupby('Prodi')['Is_Anomaly'].sum().reset_index(name='Jumlah_Anomali')
        
        audit_df = pd.merge(prodi_total, prodi_anomali, on='Prodi')
        audit_df['Tingkat_Risiko (%)'] = (audit_df['Jumlah_Anomali'] / audit_df['Total_Dosen']) * 100
        
        # Apply Filters
        audit_df = audit_df[audit_df['Jumlah_Anomali'] > 0]
        audit_df = audit_df[audit_df['Tingkat_Risiko (%)'] >= risk_threshold]
        if search_prodi:
            audit_df = audit_df[audit_df['Prodi'].str.contains(search_prodi, case=False, na=False)]
            
        audit_df = audit_df.sort_values('Tingkat_Risiko (%)', ascending=False)
        
        if audit_df.empty:
            st.success("🎉 Luar biasa! Tidak ada anomali terdeteksi di seluruh Program Studi.")
        else:
            col_a1, col_a2 = st.columns([1, 1.5])
            
            with col_a1:
                st.markdown("**Leaderboard Pengawasan**")
                st.dataframe(
                    audit_df[['Prodi', 'Total_Dosen', 'Jumlah_Anomali', 'Tingkat_Risiko (%)']],
                    column_config={
                        "Prodi": st.column_config.TextColumn("Program Studi", width="medium"),
                        "Total_Dosen": st.column_config.NumberColumn("Total", format="%d"),
                        "Jumlah_Anomali": st.column_config.NumberColumn("Anomali", format="%d"),
                        "Tingkat_Risiko (%)": st.column_config.ProgressColumn(
                            "Risiko (%)", help="Persentase dosen anomali.", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True, use_container_width=True
                )
                
            with col_a2:
                fig_risk = px.bar(
                    audit_df, x='Tingkat_Risiko (%)', y='Prodi', orientation='h',
                    color='Tingkat_Risiko (%)', color_continuous_scale=['#FCD34D', '#F59E0B', '#EF4444'],
                    text=audit_df['Tingkat_Risiko (%)'].apply(lambda x: f"{x:.1f}%")
                )
                fig_risk.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
                fig_risk.update_layout(
                    yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False,
                    font=dict(color='#E2E8F0'),
                    xaxis_title=None, yaxis_title=None, xaxis=dict(showgrid=True, gridcolor='rgba(239,68,68,0.2)', griddash='dash')
                )
                st.plotly_chart(fig_risk, use_container_width=True)

# --- TAB 2: AI PREDICTOR ---
with tab2:
    st.markdown("### Simulator Kinerja (Random Forest Model)")
    
    with st.container(border=True):
        with st.form("pred_form", clear_on_submit=False, border=False):
            st.markdown("Masukkan indikator akademis untuk diproses oleh *Artificial Intelligence*:")
            
            # Bagian 1: Identitas Afiliasi
            st.markdown("<p style='font-weight:600; color:#3B82F6; margin-bottom:5px;'>1. Identitas Afiliasi</p>", unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            prodi_list = sorted(df['Prodi'].dropna().astype(str).unique().tolist())
            fakultas_list = sorted(df['Fakultas'].dropna().astype(str).unique().tolist())
            
            selected_prodi = f_col1.selectbox("Program Studi", prodi_list)
            selected_fakultas = f_col2.selectbox("Fakultas", fakultas_list)
            
            # Bagian 2: Metrik Kinerja Utama
            st.markdown("<p style='font-weight:600; color:#3B82F6; margin-top:15px; margin-bottom:5px;'>2. Skor Kinerja SINTA</p>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            sinta_score = c1.number_input("Sinta Score (Total)", min_value=0, value=150)
            sinta_3yr = c2.number_input("Sinta Score (3 Tahun)", min_value=0, value=80)
            affil_score = c3.number_input("Skor Afiliasi", min_value=0, value=300)
            affil_3yr = c4.number_input("Skor Afiliasi (3 Thn)", min_value=0, value=120)
            
            # Bagian 3: Google Scholar
            st.markdown("<p style='font-weight:600; color:#3B82F6; margin-top:15px; margin-bottom:5px;'>3. Metrik Google Scholar</p>", unsafe_allow_html=True)
            c5, c6, c7, c8 = st.columns(4)
            gs_pub = c5.number_input("Total Publikasi GS", min_value=0, value=25)
            gs_cit = c6.number_input("Total Sitasi", min_value=0, value=150)
            gs_h = c7.number_input("H-Index", min_value=0, value=6)
            gs_i10 = c8.number_input("i10-Index", min_value=0, value=3)
            
            # Bagian 4: Riwayat Publikasi (Expander)
            st.markdown("<p style='font-weight:600; color:#3B82F6; margin-top:15px; margin-bottom:5px;'>4. Histori Publikasi (2020-2026)</p>", unsafe_allow_html=True)
            with st.expander("📊 Buka Rincian Tren Publikasi Tahunan", expanded=False):
                p_col1, p_col2, p_col3, p_col4, p_col5, p_col6, p_col7 = st.columns(7)
                p2020 = p_col1.number_input("2020", min_value=0, value=2)
                p2021 = p_col2.number_input("2021", min_value=0, value=3)
                p2022 = p_col3.number_input("2022", min_value=0, value=4)
                p2023 = p_col4.number_input("2023", min_value=0, value=4)
                p2024 = p_col5.number_input("2024", min_value=0, value=5)
                p2025 = p_col6.number_input("2025", min_value=0, value=5)
                p2026 = p_col7.number_input("2026", min_value=0, value=2)
            
            submit = st.form_submit_button("🧠 Prediksi Kinerja dengan Random Forest")
            
            if submit:
                enc_prodi = prodi_enc.transform([selected_prodi])[0]
                enc_fakultas = fakultas_enc.transform([selected_fakultas])[0]
                
                input_data = np.array([[
                    enc_prodi, enc_fakultas, 
                    sinta_score, sinta_3yr, affil_score, affil_3yr, 
                    gs_pub, gs_cit, gs_h, gs_i10,
                    p2020, p2021, p2022, p2023, p2024, p2025, p2026
                ]])
                
                pred_encoded = rf_model.predict(input_data)
                pred_class = label_encoder.inverse_transform(pred_encoded)[0]
                
                badge_class = "badge-good"
                if pred_class == "Berpengaruh": badge_class = "badge-excellent"
                elif pred_class in ["Rendah", "Buruk"]: badge_class = "badge-poor"
                elif pred_class == "Cukup": badge_class = "badge-fair"
                
                st.markdown(f"""
                <div style="margin-top: 20px; padding: 20px; border-left: 4px solid #10B981; background-color: #F0FDF4; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <p style="margin:0; font-size: 0.875rem; color: #166534; font-weight: 600;">HASIL PREDIKSI KECERDASAN BUATAN:</p>
                    <h3 style="margin:5px 0 0 0; color: #064E3B;">Kategori Kinerja: <span class="sd-badge {badge_class}" style="font-size:1rem;">{pred_class.upper()}</span></h3>
                    <p style="margin:10px 0 0 0; font-size: 0.85rem; color: #15803D;">Model *Random Forest* menganalisis 17 titik data afiliasi dan rekam jejak publikasi ini dan memberikan kesimpulan secara akurat.</p>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown("### Wawasan Algoritma (Model Insights)")
    with st.container(border=True):
        st.markdown("**Tingkat Pengaruh Fitur (Feature Importance)**")
        st.markdown("<p style='font-size:0.9rem; color:#64748B;'>Grafik di bawah ini mengungkap \"rahasia dapur\" AI: Atribut mana yang paling memengaruhi penilaian kinerja seorang dosen?</p>", unsafe_allow_html=True)
        
        feat_imp_display = feat_importance.copy()
        feat_imp_display['Fitur'] = feat_imp_display['Fitur'].str.replace('_GS', ' Google Scholar').str.replace('_', ' ')
        
        fig_imp = px.bar(feat_imp_display, x='Kepentingan', y='Fitur', orientation='h', 
                         color='Kepentingan', color_continuous_scale=['#93C5FD', '#3B82F6', '#1E3A8A'])
                         
        fig_imp.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            font=dict(color='#E2E8F0'),
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis_title=None, xaxis_title="Tingkat Pengaruh (Importance Ratio)",
            hovermode="y unified"
        )
        fig_imp.update_traces(marker_line_width=0, opacity=0.9)
        
        st.plotly_chart(fig_imp, use_container_width=True)
                
    st.markdown("### AI Peramal Publikasi (Forecasting)")
    with st.container(border=True):
        st.markdown("Model **Linear Regression** digunakan untuk membaca tren publikasi 2020-2026 dan memproyeksikan target masa depan.")
        
        # Dropdown untuk memilih scope (Universitas vs Prodi spesifik)
        prodi_list = ['Seluruh Universitas'] + sorted(df['Prodi'].dropna().unique().tolist())
        selected_scope = st.selectbox("Pilih Ruang Lingkup Analisis:", prodi_list)
        
        years_cols = ['Pub_2020', 'Pub_2021', 'Pub_2022', 'Pub_2023', 'Pub_2024', 'Pub_2025', 'Pub_2026']
        
        if selected_scope == 'Seluruh Universitas':
            hist_data = df[years_cols].sum().values
            scope_title = "Universitas Halu Oleo"
        else:
            hist_data = df[df['Prodi'] == selected_scope][years_cols].sum().values
            scope_title = f"Prodi {selected_scope}"
            
        future_data = predict_future_pubs(hist_data)
        
        st.markdown(f"**Menampilkan Proyeksi Untuk: {scope_title}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Total Publikasi 2026 (Aktual)", value=f"{hist_data[-1]:,}")
        c2.metric(label="Proyeksi Publikasi 2027", value=f"{future_data[0]:,}", delta=f"{future_data[0] - hist_data[-1]:,} dari 2026")
        c3.metric(label="Proyeksi Publikasi 2028", value=f"{future_data[1]:,}", delta=f"{future_data[1] - future_data[0]:,} dari 2027")
        
        # Plotting the forecast dengan gaya elegan
        forecast_df = pd.DataFrame({
            'Tahun': ['2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027 (Prediksi)', '2028 (Prediksi)'],
            'Jumlah Publikasi': list(hist_data) + list(future_data),
            'Tipe': ['Aktual']*7 + ['Prediksi']*2
        })
        fig_fcast = px.bar(forecast_df, x='Tahun', y='Jumlah Publikasi', color='Tipe', text='Jumlah Publikasi',
                           color_discrete_map={'Aktual': '#3B82F6', 'Prediksi': '#F59E0B'})
        fig_fcast.update_traces(texttemplate='%{text:,.0f}', textposition="outside", marker_line_width=0, cliponaxis=False)
        fig_fcast.update_layout(margin=dict(t=30, b=10, l=10, r=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                font=dict(color='#E2E8F0'),
                                legend_title_text=None, xaxis_title=None, yaxis_title=None,
                                yaxis=dict(showgrid=True, gridcolor='#334155', griddash='dash'))
        st.plotly_chart(fig_fcast, use_container_width=True)


# --- TAB 3: PROFIL DOSEN ---
with tab3:
    st.markdown("### Direktori Dosen Universitas Halu Oleo")
    search_query = st.text_input("", placeholder="🔍 Cari nama dosen (contoh: Fatma)...")
    
    if search_query:
        results = df[df['Nama_Dosen'].str.contains(search_query, case=False, na=False)]
        
        if len(results) > 0:
            st.caption(f"Menampilkan {len(results)} hasil pencarian resmi.")
            
            for _, row in results.iterrows():
                years_cols = ['Pub_2020', 'Pub_2021', 'Pub_2022', 'Pub_2023', 'Pub_2024', 'Pub_2025', 'Pub_2026']
                lecturer_hist = row[years_cols].values
                lecturer_future = predict_future_pubs(lecturer_hist)
                
                badge_class = "badge-good"
                if row['Kategori_Kinerja'] == "Berpengaruh": badge_class = "badge-excellent"
                elif row['Kategori_Kinerja'] in ["Rendah", "Buruk"]: badge_class = "badge-poor"
                elif row['Kategori_Kinerja'] == "Cukup": badge_class = "badge-fair"

                html_content = f"""
<div class="profile-header">
<div class="profile-name">{row['Nama_Dosen']}</div>
<div class="profile-meta">{row['Prodi']} — {row['Fakultas']}</div>
<span class="sd-badge {badge_class}">Machine Learning Class: {row['Kategori_Kinerja']}</span>
</div>
<div class="metric-grid">
<div class="metric-box"><div class="m-label">Sinta Score (Overall)</div><div class="m-value">{row['Sinta_Score']}</div></div>
<div class="metric-box"><div class="m-label">Sinta Score (3 Tahun)</div><div class="m-value">{row['Sinta_Score_3Yr']}</div></div>
<div class="metric-box"><div class="m-label">Total Publikasi GS</div><div class="m-value">{row['Total_Publikasi_GS']}</div></div>
<div class="metric-box"><div class="m-label">H-Index GS</div><div class="m-value">{row['h_index_GS']}</div></div>
<div class="metric-box" style="background-color: rgba(245, 158, 11, 0.15); border-color: #D97706;"><div class="m-label" style="color: #FCD34D;">🔮 Proyeksi Pub 2027</div><div class="m-value">{lecturer_future[0]}</div></div>
<div class="metric-box" style="background-color: rgba(245, 158, 11, 0.15); border-color: #D97706;"><div class="m-label" style="color: #FCD34D;">🔮 Proyeksi Pub 2028</div><div class="m-value">{lecturer_future[1]}</div></div>
</div>
"""
                with st.container(border=True):
                    st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.error("Nama tidak ditemukan di database resmi SINTA Universitas Halu Oleo.")
