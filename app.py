import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context, ALL
import io
import base64
import re
import numpy as np
from networkx.algorithms import community
import itertools
import random
from scipy.stats import pearsonr

app = Dash(__name__, title="Centrago v3.0 Expert", suppress_callback_exceptions=True)
server = app.server

# --- CONFIGURATION EXPORT IMAGE ---
PLOT_CONFIG = {
    'toImageButtonOptions': {'format': 'png', 'filename': 'centrago_export', 'height': 1080, 'width': 1920, 'scale': 2},
    'displaylogo': False,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
}

# --- DESIGN SYSTEM ---
COLORS = {
    "bg": "#0f172a", "card": "#1e293b", "accent": "#f97316", "text": "#f8fafc", 
    "border": "#334155", "clans": px.colors.qualitative.Set3, 
    "win": "#4ade80", 
    "loss": "#f87171", "map": "RdBu_r"
}

STYLE_DASH = {'backgroundColor': COLORS["bg"], 'color': COLORS["text"], 'minHeight': '100vh', 'fontFamily': 'Inter, sans-serif', 'padding': '20px'}
CARD_STYLE = {'backgroundColor': COLORS["card"], 'padding': '20px', 'borderRadius': '12px', 'marginBottom': '20px', 'border': f'1px solid {COLORS["border"]}'}
INTRO_STYLE = {**CARD_STYLE, 'borderLeft': f'6px solid {COLORS["accent"]}', 'backgroundColor': '#1e293b'}
INDICATOR_STYLE = {'textAlign': 'center', 'padding': '15px', 'borderRadius': '10px', 'backgroundColor': '#1e293b', 'border': f'1px solid {COLORS["border"]}', 'flex': 1}
BIO_TEXT_STYLE = {'color': '#94a3b8', 'fontSize': '14px', 'lineHeight': '1.6', 'marginTop': '15px', 'borderLeft': f'4px solid {COLORS["accent"]}', 'paddingLeft': '15px'}

TABLE_STYLE = {
    'style_header': {'backgroundColor': '#334155', 'color': 'white', 'fontWeight': 'bold'},
    'style_data': {'backgroundColor': COLORS["card"], 'color': COLORS["text"], 'border': '1px solid #475569'},
    'style_table': {'borderRadius': '10px', 'overflowX': 'auto'}, 'page_size': 10
}

# --- TEXTES ---
INTRO_TEXTS = {
    "tab-clans": "Analyse de la **structure communautaire** et **Test de Permutation**.",
    "tab-hier": "Analyse via **système Elo**. Classement agonistique et stabilité.",
    "tab-cent": "Mesure de l'**influence sociale**. Comparatif des indices par barplots encadrés.",
    "tab-affin": "Matrice des **préférences dyadiques** classée par rang Elo.",
    "tab-health": "Indices de **stabilité et résilience**. Analyse colorée des rôles et impact du retrait d'individus.",
    "tab-cross": "Corrélation entre les **métriques sociales et hiérarchiques**."
}

BIO_EXPLANATIONS = {
    "tab-clans": {"titre": "Analyse de la Structure", "corps": "Détection via modularité."},
    "tab-hier": {"titre": "Système de Dominance", "corps": "Le score Elo reflète la capacité de gain."},
    "tab-cent": {"titre": "Métriques de Centralité", "corps": "Comparaison du Degré, l'Intermédiarité, la Proximité et le Vecteur Propre."},
    "tab-affin": {"titre": "Matrice d'Affinité", "corps": "Axes ordonnés par dominance."},
    "tab-health": {"titre": "Cohésion & Robustesse", "corps": "SNA Stability (T vs T-1) et simulation de retrait (Impact sur la Modularité)."},
    "tab-cross": {"titre": "Synthèse Globale", "corps": "Analyse statistique multi-variée."}
}

def clean_name(nom):
    if pd.isna(nom): return "Inconnu"
    n = str(nom).strip().lower()
    return re.sub(r'^\d+\s*\.\s*', '', n).capitalize()

def make_intro_box(key):
    return html.Div([
        html.H4(f"🔍 ANALYSE : {key.replace('tab-', '').upper()}", style={'color': COLORS['accent'], 'margin': '0 0 10px 0'}),
        dcc.Markdown(INTRO_TEXTS[key], style={'color': COLORS['text'], 'fontSize': '16px'})
    ], style=INTRO_STYLE)

def make_bio_card(key):
    info = BIO_EXPLANATIONS[key]
    return html.Div([html.H5(f"📖 {info['titre']}", style={'color': COLORS['accent'], 'marginBottom': '10px'}), html.P(info['corps'], style=BIO_TEXT_STYLE)], style=CARD_STYLE)

def calculer_elo_complet(df, k=150):
    if df is None or df.empty: return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    individus = sorted(list(set(df['actor'].unique()) | set(df['receiver'].unique())))
    scores = {ind: 1200 for ind in individus}
    historique, bilan = [], {ind: {'Gains': 0, 'Pertes': 0} for ind in individus}
    for _, row in df.iterrows():
        g, p = row['actor'], row['receiver']
        if g in scores and p in scores:
            ea = 1 / (1 + 10 ** ((scores[p] - scores[g]) / 400))
            changement = k * (1 - ea)
            scores[g] += changement; scores[p] -= changement
            historique.append({'Date': row['Date'], 'Singe': g, 'Elo': scores[g]})
            historique.append({'Date': row['Date'], 'Singe': p, 'Elo': scores[p]})
            bilan[g]['Gains'] += 1; bilan[p]['Pertes'] += 1
    res = pd.DataFrame([{'Singe': ind, 'Elo_Final': round(scores[ind]), 'Gains': bilan[ind]['Gains'], 'Pertes': bilan[ind]['Pertes']} for ind in individus])
    return res.sort_values('Elo_Final', ascending=False), pd.DataFrame(historique)

def decode_df(contents):
    if contents is None: return None
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=None, engine='python')
    except: return None

def layout_centrago():
    cross_metrics = ['Elo_Final', 'Betweenness', 'Degree', 'Closeness', 'Eigenvector', 'Clustering', 'Pilier', 'Médiateur', 'Électron', 'Cible']
    return html.Div(style=STYLE_DASH, children=[
        dcc.Store(id='store-prox'), dcc.Store(id='store-agon'),
        dcc.Store(id='active-metric', data='Betweenness'), 
        html.Div([
            html.H1("🦧 CENTRAGO", style={'color': COLORS["accent"], 'margin': '0', 'fontWeight': '900', 'letterSpacing': '2px'}),
            html.P("Expertise Sociale & Éthologique Dynamique", style={'color': '#64748b'})
        ], style={'textAlign': 'center', 'marginBottom': '10px', 'marginTop': '20px'}),
        html.Div(id='global-date-container', style={'marginBottom': '20px'}),
        html.Div(id='cross-controls-container', style={'display': 'none'}, children=[
            html.Div(style=CARD_STYLE, children=[
                html.H5("🧪 PARAMÈTRES DES AXES", style={'color': COLORS["accent"], 'marginTop': '0'}),
                html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                    html.Div([html.Label("Axe X"), dcc.Dropdown(id='xaxis-var', options=[{'label': k, 'value': k} for k in cross_metrics], value='Elo_Final', style={'color': 'black'})], style={'flex': 1}),
                    html.Div([html.Label("Axe Y"), dcc.Dropdown(id='yaxis-var', options=[{'label': k, 'value': k} for k in cross_metrics], value='Betweenness', style={'color': 'black'})], style={'flex': 1}),
                ])
            ])
        ]),
        dcc.Tabs(id="tabs", value='tab-clans', children=[
            dcc.Tab(label='👥 CLANS (3D)', value='tab-clans'),
            dcc.Tab(label='👑 HIÉRARCHIE', value='tab-hier'),
            dcc.Tab(label='📊 CENTRALITÉ', value='tab-cent'),
            dcc.Tab(label='🤝 AFFINITÉS', value='tab-affin'),
            dcc.Tab(label='🧬 COHÉSION', value='tab-health'),
            dcc.Tab(label='🧪 ANALYSE CROISÉE', value='tab-cross'),
        ]),
        html.Div(id='tab-content', style={'marginTop': '25px'})
    ])

app.layout = html.Div([dcc.Location(id='url', refresh=False), html.Div(id='page-content')])

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname): return layout_centrago()

@app.callback(Output('cross-controls-container', 'style'), [Input('tabs', 'value')])
def show_controls(tab): return {'display': 'block'} if tab == 'tab-cross' else {'display': 'none'}

@app.callback(Output('store-prox', 'data'), Input('upload-prox', 'contents'))
def update_p(c): return c

@app.callback(Output('store-agon', 'data'), Input('upload-agon', 'contents'))
def update_a(c): return c

@app.callback(Output('active-metric', 'data'), [Input('metric-selector', 'value')], prevent_initial_call=True)
def update_metric(v): return v if v else 'Betweenness'

@app.callback(Output('global-date-container', 'children'), [Input('store-prox', 'data')])
def render_global_slider(prox_c):
    if not prox_c: return None
    dfp = decode_df(prox_c)
    if dfp is None or 'Date' not in dfp.columns: return None
    dfp['Date'] = pd.to_datetime(dfp['Date'], dayfirst=True, errors='coerce')
    unique_dates = sorted(dfp['Date'].dropna().unique())
    marks = {i: {'label': d.strftime('%d/%m'), 'style': {'color': COLORS['text'], 'fontSize': '10px'}} for i, d in enumerate(unique_dates)}
    return html.Div([
        html.Div([html.Span("⏱️ NAVIGATION TEMPORELLE", style={'color': COLORS['accent'], 'fontWeight': 'bold'})], style={'marginBottom': '10px'}),
        html.Div([
            html.Div([dcc.Slider(id={'type': 'date-slider', 'index': 'global'}, min=0, max=len(unique_dates), value=len(unique_dates), marks={**marks, len(unique_dates): {'label': '📊 TOTAL', 'style': {'color': COLORS['accent'], 'fontWeight': 'bold'}}}, step=1)], style={'flex': '1', 'paddingRight': '20px'}),
            html.Button("RÉINITIALISER", id="btn-set-total", style={'backgroundColor': COLORS['accent'], 'border': 'none', 'color': 'white', 'padding': '10px', 'borderRadius': '8px', 'cursor': 'pointer', 'fontWeight': 'bold'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={**CARD_STYLE, 'borderBottom': f'2px solid {COLORS["accent"]}'})

@app.callback(Output({'type': 'date-slider', 'index': 'global'}, 'value'), Input('btn-set-total', 'n_clicks'), State({'type': 'date-slider', 'index': 'global'}, 'max'), prevent_initial_call=True)
def force_total(n, max_val): return max_val

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'), Input('store-prox', 'data'), Input('store-agon', 'data'),
     Input('xaxis-var', 'value'), Input('yaxis-var', 'value'), Input('active-metric', 'data'),
     Input({'type': 'date-slider', 'index': ALL}, 'value'),
     Input({'type': 'radar-subject', 'index': ALL}, 'value')]
)
def render_master(tab, prox_c, agon_c, x_var, y_var, active_m, slider_val, radar_sub):
    needs_prox = tab in ['tab-clans', 'tab-cent', 'tab-affin', 'tab-health', 'tab-cross']
    needs_agon = tab in ['tab-hier', 'tab-affin', 'tab-health', 'tab-cross']
    if needs_prox and prox_c is None: return html.Div([dcc.Upload(id='upload-prox', children=html.Div(['📂 Charger PROXIMITÉ']), style={'height': '100px', 'lineHeight': '100px', 'border': '2px dashed #475569', 'textAlign': 'center'})], style=CARD_STYLE)
    if needs_agon and agon_c is None: return html.Div([dcc.Upload(id='upload-agon', children=html.Div(['📂 Charger AGONISTIQUE']), style={'height': '100px', 'lineHeight': '100px', 'border': '2px dashed #475569', 'textAlign': 'center'})], style=CARD_STYLE)
    
    try:
        dfp_raw = decode_df(prox_c); dfa = decode_df(agon_c)
        dfp = dfp_raw.copy()
        if slider_val and dfp is not None:
            dfp['Date'] = pd.to_datetime(dfp['Date'], dayfirst=True, errors='coerce')
            unique_dates = sorted(dfp['Date'].dropna().unique())
            idx = slider_val[0]
            if idx < len(unique_dates):
                sel_date = unique_dates[idx]; dfp = dfp[dfp['Date'] == sel_date]
                if dfa is not None: 
                    dfa['Date'] = pd.to_datetime(dfa['Date'], dayfirst=True, errors='coerce'); dfa = dfa[dfa['Date'] <= sel_date]

        if dfp is not None:
            dfp['Subject'] = dfp['Subject'].apply(clean_name); dfp['Partner'] = dfp['Partner'].apply(clean_name)
        if dfa is not None:
            dfa['actor'] = dfa['actor'].apply(clean_name); dfa['receiver'] = dfa['receiver'].apply(clean_name)
        if dfp is not None:
            edges_df = dfp[dfp['Partner'] != "Alone"].groupby(['Subject', 'Partner']).size().reset_index(name='weight')
            G = nx.from_pandas_edgelist(edges_df, 'Subject', 'Partner', ['weight'])
        
        if tab == 'tab-clans':
            communities = list(community.greedy_modularity_communities(G, weight='weight'))
            clans_dict = {node: f"Clan {i+1}" for i, comm in enumerate(communities) for node in comm}
            df_clans = pd.DataFrame(list(clans_dict.items()), columns=['Singe', 'Clan']).sort_values('Clan')
            mod_obs = community.modularity(G, communities, weight='weight')
            null_mods = []
            nodes_list = list(G.nodes())
            for _ in range(100):
                perm_labels = nodes_list.copy(); random.shuffle(perm_labels)
                label_map = dict(zip(nodes_list, perm_labels)); G_perm = nx.relabel_nodes(G, label_map)
                c_perm = list(community.greedy_modularity_communities(G_perm, weight='weight'))
                null_mods.append(community.modularity(G_perm, c_perm, weight='weight'))
            p_value = np.mean([m >= mod_obs for m in null_mods])
            fig_perm = px.histogram(null_mods, nbins=20, template='plotly_dark', title="Distribution de Modularité", color_discrete_sequence=['#94a3b8'])
            fig_perm.add_vline(x=mod_obs, line_width=3, line_dash="dash", line_color=COLORS['accent'])
            pos_3d = nx.spring_layout(G, dim=3, seed=42); fig_3d = go.Figure()
            for edge in G.edges(data=True):
                x0, y0, z0 = pos_3d[edge[0]]; x1, y1, z1 = pos_3d[edge[1]]
                fig_3d.add_trace(go.Scatter3d(x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None], mode='lines', line=dict(color='white', width=2), hoverinfo='none', opacity=0.4, showlegend=False))
            for i, comm in enumerate(communities): 
                fig_3d.add_trace(go.Scatter3d(x=[pos_3d[n][0] for n in comm], y=[pos_3d[n][1] for n in comm], z=[pos_3d[n][2] for n in comm], mode='markers+text', name=f"Clan {i+1}", text=list(comm), marker=dict(size=15, color=COLORS['clans'][i % len(COLORS['clans'])])))
            return html.Div([make_intro_box(tab), html.Div([html.Div([html.Small("MODULARITÉ"), html.H3(f"{mod_obs:.3f}")], style=INDICATOR_STYLE), html.Div([html.Small("P-VALUE"), html.H3(f"{p_value:.3f}", style={'color': COLORS['win'] if p_value < 0.05 else COLORS['loss']})], style=INDICATOR_STYLE)], style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px'}), html.Div([html.Div([dash_table.DataTable(data=df_clans.to_dict('records'), **TABLE_STYLE)], style={'flex': 1}), html.Div([dcc.Graph(figure=fig_perm, config=PLOT_CONFIG)], style={'flex': 1.5})], style={'display': 'flex', 'gap': '15px', **CARD_STYLE}), html.Div([dcc.Graph(figure=fig_3d.update_layout(template='plotly_dark', margin=dict(l=0,r=0,b=0,t=0)), config=PLOT_CONFIG, style={'height': '70vh'})], style=CARD_STYLE), make_bio_card(tab)])

        elif tab == 'tab-hier':
            df_s, df_h = calculer_elo_complet(dfa); df_h_pivot = df_h.pivot_table(index='Date', columns='Singe', values='Elo', aggfunc='last').ffill()
            ranks = df_h_pivot.rank(axis=1, ascending=False); turbulence = ranks.std().mean() if len(df_h_pivot) > 1 else 0.0
            if not df_s.empty:
                max_elo = df_s['Elo_Final'].max(); min_elo = df_s['Elo_Final'].min()
                steepness = (max_elo - min_elo) / max_elo if max_elo > 0 else 0
                df_s_norm = df_s.sort_values('Elo_Final', ascending=False).reset_index(drop=True); df_s_norm['Rang'] = df_s_norm.index + 1
                fig_steep = px.scatter(df_s_norm, x='Rang', y='Elo_Final', text='Singe', trendline="ols", template='plotly_dark', title="Pente Elo")
                fig_steep.update_traces(marker=dict(size=12, color=COLORS['accent']), textposition='top center')
            else: steepness = 0; fig_steep = go.Figure()
            fig_ratio = px.bar(df_s, x='Singe', y=['Gains', 'Pertes'], barmode='group', template='plotly_dark', color_discrete_map={'Gains': COLORS['win'], 'Pertes': COLORS['loss']})
            fig_evol = px.line(df_h, x='Date', y='Elo', color='Singe', template='plotly_dark')
            return html.Div([make_intro_box(tab), html.Div([html.Div([html.Small("TURBULENCE"), html.H3(f"{turbulence:.2f}")], style=INDICATOR_STYLE), html.Div([html.Small("STEEPNESS"), html.H3(f"{steepness:.2f}")], style=INDICATOR_STYLE)], style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px'}), html.Div([dash_table.DataTable(data=df_s.to_dict('records'), **TABLE_STYLE)], style=CARD_STYLE), html.Div([dcc.Graph(figure=fig_steep, config=PLOT_CONFIG)], style=CARD_STYLE), html.Div([dcc.Graph(figure=fig_ratio, config=PLOT_CONFIG)], style=CARD_STYLE), html.Div([dcc.Graph(figure=fig_evol, config=PLOT_CONFIG)], style=CARD_STYLE), make_bio_card(tab)])

        elif tab == 'tab-cent':
            bet = nx.betweenness_centrality(G, weight='weight'); deg = dict(G.degree(weight='weight'))
            clo = nx.closeness_centrality(G, distance='weight'); clus = nx.clustering(G, weight='weight')
            try: eig = nx.eigenvector_centrality_numpy(G, weight='weight')
            except: eig = {n: 0 for n in G.nodes()}
            cent_df = pd.DataFrame({'Singe': list(G.nodes()), 'Degree': [deg[n] for n in G.nodes()], 'Betweenness': [round(bet[n], 3) for n in G.nodes()], 'Closeness': [round(clo[n], 3) for n in G.nodes()], 'Eigenvector': [round(eig[n], 3) for n in G.nodes()], 'Clustering': [round(clus[n], 3) for n in G.nodes()]})
            fig_bar_deg = px.bar(cent_df.sort_values('Degree', ascending=False), x='Singe', y='Degree', template='plotly_dark', color='Degree', color_continuous_scale='RdBu_r', title="Degree")
            fig_bar_bet = px.bar(cent_df.sort_values('Betweenness', ascending=False), x='Singe', y='Betweenness', template='plotly_dark', color='Betweenness', color_continuous_scale='RdBu_r', title="Betweenness")
            fig_bar_clo = px.bar(cent_df.sort_values('Closeness', ascending=False), x='Singe', y='Closeness', template='plotly_dark', color='Closeness', color_continuous_scale='RdBu_r', title="Closeness")
            fig_bar_eig = px.bar(cent_df.sort_values('Eigenvector', ascending=False), x='Singe', y='Eigenvector', template='plotly_dark', color='Eigenvector', color_continuous_scale='RdBu_r', title="Eigenvector")
            
            barplots_container = html.Div([
                html.H5("📊 COMPARAISON DES INDICES", style={'color': COLORS['accent'], 'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div([html.Div([dcc.Graph(figure=fig_bar_deg, config=PLOT_CONFIG)], style={'flex': 1}), html.Div([dcc.Graph(figure=fig_bar_bet, config=PLOT_CONFIG)], style={'flex': 1})], style={'display': 'flex', 'gap': '15px', 'marginBottom': '15px'}),
                html.Div([html.Div([dcc.Graph(figure=fig_bar_clo, config=PLOT_CONFIG)], style={'flex': 1}), html.Div([dcc.Graph(figure=fig_bar_eig, config=PLOT_CONFIG)], style={'flex': 1})], style={'display': 'flex', 'gap': '15px'})
            ], style={**CARD_STYLE, 'border': f'2px solid {COLORS["accent"]}', 'backgroundColor': '#111827'})
            
            node_colors = cent_df[active_m if active_m in cent_df.columns else 'Betweenness']
            pos_2d = nx.spring_layout(G, seed=42); fig_2d = go.Figure()
            max_w = max([d['weight'] for u, v, d in G.edges(data=True)]) if G.edges() else 1
            for u, v, d in G.edges(data=True):
                x0, y0 = pos_2d[u]; x1, y1 = pos_2d[v]; w = (d['weight'] / max_w) * 5 + 0.5
                fig_2d.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(color='white', width=w), hoverinfo='none', opacity=0.4, showlegend=False))
            fig_2d.add_trace(go.Scatter(x=[pos_2d[n][0] for n in G.nodes()], y=[pos_2d[n][1] for n in G.nodes()], mode='markers+text', text=list(G.nodes()), marker=dict(size=25, color=node_colors, colorscale='RdBu_r', showscale=True, colorbar=dict(title=active_m)), showlegend=False))
            pos_3d = nx.spring_layout(G, dim=3, seed=42); fig_3d = go.Figure()
            for u, v, d in G.edges(data=True):
                x0, y0, z0 = pos_3d[u]; x1, y1, z1 = pos_3d[v]; w = (d['weight'] / max_w) * 8 + 1
                fig_3d.add_trace(go.Scatter3d(x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None], mode='lines', line=dict(color='white', width=w), hoverinfo='none', opacity=0.3, showlegend=False))
            fig_3d.add_trace(go.Scatter3d(x=[pos_3d[n][0] for n in G.nodes()], y=[pos_3d[n][1] for n in G.nodes()], z=[pos_3d[n][2] for n in G.nodes()], mode='markers+text', text=list(G.nodes()), marker=dict(size=8, color=node_colors, colorscale='RdBu_r'), showlegend=False))
            subj = radar_sub[0] if radar_sub else cent_df['Singe'].iloc[0]
            row = cent_df[cent_df['Singe'] == subj].iloc[0]; metrics = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'Clustering']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[cent_df[m].mean()/cent_df[m].max() if cent_df[m].max()!=0 else 0 for m in metrics], theta=metrics, fill='toself', name='Moyenne', line_color='#94a3b8', opacity=0.5))
            fig_radar.add_trace(go.Scatterpolar(r=[row[m]/cent_df[m].max() if cent_df[m].max()!=0 else 0 for m in metrics], theta=metrics, fill='toself', name=subj, line_color=COLORS['accent']))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template='plotly_dark')
            return html.Div([make_intro_box(tab), html.Div([dash_table.DataTable(data=cent_df.sort_values(active_m, ascending=False).to_dict('records'), **TABLE_STYLE)], style=CARD_STYLE), barplots_container, html.Div([html.Div([html.Label("Singe :"), dcc.Dropdown(id={'type': 'radar-subject', 'index': 'radar'}, options=[{'label': n, 'value': n} for n in cent_df['Singe']], value=subj, style={'color': 'black'}), dcc.Graph(figure=fig_radar, config=PLOT_CONFIG)], style={'flex': 1}), html.Div([html.Label("Métrique :"), dcc.Dropdown(id='metric-selector', options=[{'label': k, 'value': k} for k in ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'Clustering']], value=active_m, style={'color': 'black'}), dcc.Graph(figure=fig_2d.update_layout(template='plotly_dark', margin=dict(l=0,r=0,b=0,t=40)), config=PLOT_CONFIG)], style={'flex': 1.2})], style={'display': 'flex', 'gap': '15px', **CARD_STYLE}), html.Div([dcc.Graph(figure=fig_3d.update_layout(template='plotly_dark', margin=dict(l=0,r=0,b=0,t=40)), config=PLOT_CONFIG, style={'height': '60vh'})], style=CARD_STYLE), make_bio_card(tab)])

        elif tab == 'tab-affin':
            df_s, _ = calculer_elo_complet(dfa); ordre_hier = df_s['Singe'].tolist()
            dyades = dfp[dfp['Partner'] != "Alone"].groupby(['Subject', 'Partner']).size().reset_index(name='Freq')
            fig = px.density_heatmap(dyades, x='Subject', y='Partner', z='Freq', template='plotly_dark', color_continuous_scale="RdBu_r", category_orders={'Subject': ordre_hier, 'Partner': ordre_hier})
            return html.Div([make_intro_box(tab), html.Div([dash_table.DataTable(data=dyades.sort_values('Freq', ascending=False).head(20).to_dict('records'), **TABLE_STYLE)], style=CARD_STYLE), html.Div([dcc.Graph(figure=fig, config=PLOT_CONFIG)], style=CARD_STYLE), make_bio_card(tab)])

        elif tab == 'tab-health':
            densite = nx.density(G); transitivite = nx.transitivity(G)
            
            # CALCUL SNA STABILITY
            sna_stability = 0.0
            if slider_val and dfp_raw is not None:
                dfp_raw['Date'] = pd.to_datetime(dfp_raw['Date'], dayfirst=True, errors='coerce')
                u_dates = sorted(dfp_raw['Date'].dropna().unique())
                idx = slider_val[0]
                if idx > 0 and idx < len(u_dates):
                    d_cur = u_dates[idx]; d_prev = u_dates[idx-1]
                    df_c = dfp_raw[dfp_raw['Date'] == d_cur]; df_p = dfp_raw[dfp_raw['Date'] == d_prev]
                    g_c = nx.from_pandas_edgelist(df_c[df_c['Partner'] != "Alone"].groupby(['Subject', 'Partner']).size().reset_index(name='w'), 'Subject', 'Partner', 'w')
                    g_p = nx.from_pandas_edgelist(df_p[df_p['Partner'] != "Alone"].groupby(['Subject', 'Partner']).size().reset_index(name='w'), 'Subject', 'Partner', 'w')
                    common_nodes = list(set(g_c.nodes()) & set(g_p.nodes()))
                    if len(common_nodes) > 1:
                        deg_c = [dict(g_c.degree(weight='w'))[n] for n in common_nodes]
                        deg_p = [dict(g_p.degree(weight='w'))[n] for n in common_nodes]
                        sna_stability, _ = pearsonr(deg_c, deg_p)

            # CALCUL ROBUSTESSE (IMPACT DU RETRAIT)
            nodes = list(G.nodes())
            robustness_data = []
            orig_comm = list(community.greedy_modularity_communities(G, weight='weight'))
            orig_mod = community.modularity(G, orig_comm, weight='weight')
            
            for node in nodes:
                G_temp = G.copy(); G_temp.remove_node(node)
                if G_temp.number_of_nodes() > 1:
                    comm_temp = list(community.greedy_modularity_communities(G_temp, weight='weight'))
                    mod_temp = community.modularity(G_temp, comm_temp, weight='weight')
                    robustness_data.append({'Singe': node, 'Impact_Modularité': mod_temp - orig_mod})
            
            df_rob = pd.DataFrame(robustness_data).sort_values('Impact_Modularité')
            fig_robust = px.bar(df_rob, x='Singe', y='Impact_Modularité', 
                               title="Robustesse : Quel singe maintient les clans ?", 
                               template='plotly_dark', color='Impact_Modularité', color_continuous_scale='RdYlGn')

            def shannon(node):
                weights = [d['weight'] for u, v, d in G.edges(node, data=True)]
                if not weights: return 0
                prob = np.array(weights) / sum(weights); return -np.sum(prob * np.log(prob + 1e-9))
            div_sociale = np.mean([shannon(n) for n in G.nodes()])
            TRIAD_LABELS = {'003': 'Trio sans lien', '102': 'Paire unique', '201': 'Chaîne', '300': 'Clique'}
            triades_raw = nx.triadic_census(G.to_directed())
            df_tri_pie = pd.DataFrame([{'Motif': TRIAD_LABELS.get(k, k), 'Nombre': v} for k, v in triades_raw.items() if v > 0 and k in TRIAD_LABELS])
            fig_triades = px.pie(df_tri_pie, values='Nombre', names='Motif', title="Triades", template='plotly_dark')
            
            role_data = {n: {'Pilier (300)': 0, 'Médiateur (201)': 0, 'Électron (201)': 0, 'Cible (102)': 0} for n in nodes}
            for trio in itertools.combinations(nodes, 3):
                sub = G.subgraph(trio); e = sub.number_of_edges()
                if e == 3: 
                    for n in trio: role_data[n]['Pilier (300)'] += 1
                elif e == 2:
                    for n in trio:
                        if sub.degree(n) == 2: role_data[n]['Médiateur (201)'] += 1
                        else: role_data[n]['Électron (201)'] += 1
                elif e == 1:
                    for n in trio:
                        if sub.degree(n) == 0: role_data[n]['Cible (102)'] += 1
            df_roles = pd.DataFrame.from_dict(role_data, orient='index').reset_index().rename(columns={'index': 'Singe'})
            
            def get_status(row):
                if row['Pilier (300)'] > 1: return "Noyau Dur"
                if row['Médiateur (201)'] > 1: return "Pivot"
                if row['Électron (201)'] > 1: return "Électron Libre"
                return "Périphérique"
            df_roles['Statut Dominant'] = df_roles.apply(get_status, axis=1)
            
            style_data_conditional = [
                {'if': {'column_id': 'Statut Dominant', 'filter_query': '{Statut Dominant} eq "Noyau Dur"'}, 'backgroundColor': '#1e3a8a', 'color': 'white'},
                {'if': {'column_id': 'Statut Dominant', 'filter_query': '{Statut Dominant} eq "Pivot"'}, 'backgroundColor': '#14532d', 'color': 'white'},
                {'if': {'column_id': 'Statut Dominant', 'filter_query': '{Statut Dominant} eq "Électron Libre"'}, 'backgroundColor': '#78350f', 'color': 'white'},
                {'if': {'column_id': 'Statut Dominant', 'filter_query': '{Statut Dominant} eq "Périphérique"'}, 'backgroundColor': '#451a03', 'color': 'white'}
            ]

            df_s, _ = calculer_elo_complet(dfa); steep = (df_s['Elo_Final'].max() - df_s['Elo_Final'].min()) / df_s['Elo_Final'].max() if not df_s.empty else 0
            fig_coh = go.Figure()
            fig_coh.add_trace(go.Scatterpolar(r=[densite, transitivite, steep, div_sociale/2], theta=['Densité', 'Transitivité', 'Steepness', 'Diversité'], fill='toself', line_color='#f472b6'))
            fig_coh.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template='plotly_dark')
            
            return html.Div([
                make_intro_box(tab), 
                html.Div([
                    html.Div([html.Small("DENSITÉ"), html.H3(f"{densite:.2%}")], style=INDICATOR_STYLE), 
                    html.Div([html.Small("SNA STABILITY (T vs T-1)"), html.H3(f"{sna_stability:.3f}", style={'color': COLORS['accent']})], style=INDICATOR_STYLE), 
                    html.Div([html.Small("DIVERSITÉ"), html.H3(f"{div_sociale:.2f}")], style=INDICATOR_STYLE)
                ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px'}), 
                html.Div([html.Div([dcc.Graph(figure=fig_coh, config=PLOT_CONFIG)], style={'flex': 1}), html.Div([dcc.Graph(figure=fig_triades, config=PLOT_CONFIG)], style={'flex': 1})], style={'display': 'flex', 'gap': '15px', **CARD_STYLE}), 
                html.Div([
                    dcc.Graph(figure=fig_robust, config=PLOT_CONFIG),
                    html.Div([
                        html.P("🛑 Rouge : Le mur du groupe (si retiré, la structure clans s'écroule).", style={'color': COLORS['loss'], 'fontSize': '13px'}),
                        html.P("✅ Vert : La porte du groupe (si retiré, le groupe se casse en clans isolés).", style={'color': COLORS['win'], 'fontSize': '13px'})
                    ], style={'padding': '10px'})
                ], style=CARD_STYLE),
                html.Div([html.H5("💎 RÔLES TRIADIQUES", style={'color': COLORS['accent']}), dash_table.DataTable(data=df_roles.to_dict('records'), style_data_conditional=style_data_conditional, **TABLE_STYLE)], style=CARD_STYLE),
                make_bio_card(tab)
            ])

        elif tab == 'tab-cross':
            df_s, _ = calculer_elo_complet(dfa); bet = nx.betweenness_centrality(G, weight='weight'); deg = dict(G.degree(weight='weight'))
            clo = nx.closeness_centrality(G, distance='weight'); eig = nx.eigenvector_centrality_numpy(G, weight='weight')
            net_df = pd.DataFrame({'Singe': list(G.nodes()), 'Betweenness': [round(bet[n], 3) for n in G.nodes()], 'Degree': [deg[n] for n in G.nodes()], 'Closeness': [round(clo[n], 3) for n in G.nodes()], 'Eigenvector': [round(eig[n], 3) for n in G.nodes()]})
            df_final = pd.merge(net_df, df_s[['Singe', 'Elo_Final']], on='Singe')
            fig = px.scatter(df_final, x=x_var, y=y_var, size='Degree', text='Singe', template='plotly_dark', color='Elo_Final', color_continuous_scale="RdBu_r", trendline="ols")
            return html.Div([make_intro_box(tab), html.Div([dash_table.DataTable(data=df_final.to_dict('records'), **TABLE_STYLE)], style=CARD_STYLE), html.Div([dcc.Graph(figure=fig, config=PLOT_CONFIG)], style=CARD_STYLE), make_bio_card(tab)])
    
    except Exception as e: return html.Div([html.H5(f"⚠️ Erreur : {str(e)}", style={'color': '#ef4444'})], style=CARD_STYLE)

if __name__ == '__main__': app.run(debug=True)
