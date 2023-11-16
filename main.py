import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# Carregar o DataFrame
df = pd.read_excel("df_merged.xlsx")

# Calculate correlations
correlations = []
years = list(range(1995, 2021))

for year in years:
    hdi = f"hdi_{year}"
    ef = f"ef_{year}"

    select_columns = ["iso3", "country", hdi, ef]
    df_year = df[select_columns].dropna()

    # Cálculo dos rankings
    df_year[f"{hdi}_rank"] = df_year[hdi].rank(ascending=False)
    df_year[f"{ef}_rank"] = df_year[ef].rank(ascending=False)

    # Ordena o DataFrame pelos rankings
    df_year = df_year.sort_values(by=[f"{hdi}_rank", f"{ef}_rank"])

    # Cálculo da correlação
    correlation = df_year[[f"{hdi}_rank", f"{ef}_rank"]].corr(method="pearson")
    correlations.append(correlation.iloc[0, 1])

    # Adiciona uma lista de códigos ISO como opções para o dropdown
    iso_options = [{"label": iso, "value": iso} for iso in df["iso3"].unique()]


def criar_box_plot():
    # Leitura do DataFrame
    df = pd.read_excel("df_idh_filtrado.xlsx")

    # Seleção dos dados de 2021
    idh_2021 = df[["iso3", "country", "hdi_2021"]]

    # Ordenação por HDI
    idh_2021_sorted = idh_2021.sort_values(by="hdi_2021", ascending=False)

    # TOP 20
    top_20_idh = idh_2021_sorted.head(20)

    # G20
    g20_countries = [
        "ARG",
        "AUS",
        "BRA",
        "CAN",
        "CHN",
        "FRA",
        "DEU",
        "IND",
        "IDN",
        "ITA",
        "JPN",
        "MEX",
        "RUS",
        "SAU",
        "ZAF",
        "KOR",
        "TUR",
        "GBR",
        "USA",
    ]
    g20_df = idh_2021[idh_2021["iso3"].isin(g20_countries)]

    # Criação do Box Plot
    box_plot_data = [
        g20_df["hdi_2021"].values,
        top_20_idh["hdi_2021"].values,
        idh_2021_sorted["hdi_2021"].values,
    ]

    # Criar uma figura Plotly diretamente
    fig = px.box(
        pd.DataFrame(box_plot_data).transpose(),
        points="all",
        labels={"variable": "Grupo", "value": "IDH"},
    )

    fig.update_layout(
        title="Box Plots do IDH dos Países do G20, Top 20 e Mundo em 2021",
        yaxis=dict(title="IDH (Índice de Desenvolvimento Humano)"),
        showlegend=False,
    )

    # Atualizar rótulos no eixo x
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=["G20", "Top 20", "Mundo"])

    return fig


# Função para carregar os DataFrames e calcular as correlações
def load_and_calculate_correlations():
    # Carregar os DataFrames
    merged_df_fraser = pd.read_csv("merged_df_fraser.csv")
    merged_df_heritage = pd.read_csv("merged_df_heritage.csv")
    merged_df_fraser_heritage = pd.read_csv("merged_df_fraser_heritage.csv")

    # Calcular as correlações
    correlation_rankings_fraser = merged_df_fraser[
        ["Ranking HDI 2020", "Ranking Economic Freedom"]
    ].corr(method="pearson")
    correlation_hdi_ef_fraser = merged_df_fraser[
        ["hdi_2020", "Economic Freedom Summary Index"]
    ].corr(method="pearson")

    correlation_rankings_heritage = merged_df_heritage[
        ["Ranking HDI 2020", "Ranking Economic Freedom"]
    ].corr(method="pearson")
    correlation_hdi_ef_heritage = merged_df_heritage[
        ["hdi_2020", "Overall Score"]
    ].corr(method="pearson")

    correlation_rankings_fraser_heritage = merged_df_fraser_heritage[
        ["Ranking Economic Freedom_x", "Ranking Economic Freedom_y"]
    ].corr(method="pearson")
    correlation_index_fraser_heritage = merged_df_fraser_heritage[
        ["Economic Freedom Summary Index", "Overall Score"]
    ].corr(method="pearson")

    return (
        merged_df_fraser,
        merged_df_heritage,
        merged_df_fraser_heritage,
        correlation_rankings_fraser,
        correlation_hdi_ef_fraser,
        correlation_rankings_heritage,
        correlation_hdi_ef_heritage,
        correlation_rankings_fraser_heritage,
        correlation_index_fraser_heritage,
    )


# Carregar e calcular as correlações
(
    merged_df_fraser,
    merged_df_heritage,
    merged_df_fraser_heritage,
    correlation_rankings_fraser,
    correlation_hdi_ef_fraser,
    correlation_rankings_heritage,
    correlation_hdi_ef_heritage,
    correlation_rankings_fraser_heritage,
    correlation_index_fraser_heritage,
) = load_and_calculate_correlations()


# Define Dash app
app = dash.Dash(__name__)

# Título Geral e Descrição
titulo_geral = "Análise de Correlação: Desenvolvimento Humano e Liberdade Econômica"
descricao = [
    "Este projeto de estatística visa analisar a correlação entre o ",
    html.Strong("Índice de Desenvolvimento Humano (HDI)"),
    " e o ",
    html.Strong("Índice de Liberdade Econômica (EFI)"),
    ". Os dados utilizados foram obtidos do ",
    html.Strong("Fraser Institute"),
    ", ",
    html.Strong("Heritage Foundation"),
    " e ",
    html.Strong("United Nations Development Programme (UNDP)"),
    "."
]

# layout
app.layout = html.Div(
    children=[
        # Título Geral e Descrição
        html.Div(
            [
                html.H1(children=titulo_geral, className="titulo-geral"),
                html.Hr(className="linha-separadora"),
                html.P(children=descricao, className="descricao"),
            ],
            className="titulo-descricao-container",
        ),
        # # Div Fraser Dataframe vs HDI 2020
        # html.Div(
        #     children=[
        #         html.H2("Fraser Dataframe vs HDI 2020:", className="title-graphs"),
        #         html.Hr(className="linha-separadora"),
        #         html.P("Correlation between Rankings:"),
        #         html.Pre(children=str(correlation_rankings_fraser)),
        #         html.P("Correlation between HDI 2020 and Economic Freedom:"),
        #         html.Pre(children=str(correlation_hdi_ef_fraser)),
        #     ],
        #     className="corr-container",
        # ),
        # # Div Heritage Dataframe vs HDI 2020
        # html.Div(
        #     children=[
        #         html.H2("Heritage Dataframe vs HDI 2020:", className="title-graphs"),
        #         html.Hr(className="linha-separadora"),
        #         html.P("Correlation between Rankings:"),
        #         html.Pre(children=str(correlation_rankings_heritage)),
        #         html.P("Correlation between HDI 2020 and Economic Freedom:"),
        #         html.Pre(children=str(correlation_hdi_ef_heritage)),
        #     ],
        #     className="corr-container",
        # ),
        # # Div Fraser Dataframe vs Heritage Dataframe
        # html.Div(
        #     children=[
        #         html.H2(
        #             "Fraser Dataframe vs Heritage Dataframe:", className="title-graphs"
        #         ),
        #         html.Hr(className="linha-separadora"),
        #         html.P("Correlation between Rankings:"),
        #         html.Pre(children=str(correlation_rankings_fraser_heritage)),
        #         html.P(
        #             "Correlation between Economic Freedom Summary Index and Overall Score:"
        #         ),
        #         html.Pre(children=str(correlation_index_fraser_heritage)),
        #     ],
        #     className="corr-container",
        # ),
        # Div 1 Fraser Dataframe vs HDI 2020
        html.Div(
            children=[
                html.H2("Fraser Dataframe vs HDI 2020:", className="title-graphs"),
                html.Hr(className="linha-separadora"),
                html.P("Correlation between Rankings:"),
                # Tabela 1
                dash_table.DataTable(
                    columns=[
                        {"name": "", "id": "index"},
                        {"name": "Ranking HDI 2020", "id": "Ranking HDI 2020"},
                        {
                            "name": "Ranking Economic Freedom",
                            "id": "Ranking Economic Freedom",
                        },
                    ],
                    data=[
                        {
                            "index": "Ranking HDI 2020",
                            "Ranking HDI 2020": correlation_rankings_fraser.iloc[0, 0],
                            "Ranking Economic Freedom": correlation_rankings_fraser.iloc[
                                0, 1
                            ],
                        },
                        {
                            "index": "Ranking Economic Freedom",
                            "Ranking HDI 2020": correlation_rankings_fraser.iloc[1, 0],
                            "Ranking Economic Freedom": correlation_rankings_fraser.iloc[
                                1, 1
                            ],
                        },
                    ],
                    style_table={
                        "maxWidth": "80%",
                        "margin": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"textAlign": "center"},
                ),
                html.P("Correlation between HDI 2020 and Economic Freedom:"),
                # Tabela 2
                dash_table.DataTable(
                    columns=[
                        {"name": "", "id": "index"},
                        {"name": "hdi_2020", "id": "hdi_2020"},
                        {
                            "name": "Economic Freedom Summary Index",
                            "id": "Economic Freedom Summary Index",
                        },
                    ],
                    data=[
                        {
                            "index": "hdi_2020",
                            "hdi_2020": correlation_hdi_ef_fraser.iloc[0, 0],
                            "Economic Freedom Summary Index": correlation_hdi_ef_fraser.iloc[
                                0, 1
                            ],
                        },
                        {
                            "index": "Economic Freedom Summary Index",
                            "hdi_2020": correlation_hdi_ef_fraser.iloc[1, 0],
                            "Economic Freedom Summary Index": correlation_hdi_ef_fraser.iloc[
                                1, 1
                            ],
                        },
                    ],
                    style_table={
                        "maxWidth": "80%",
                        "margin": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"textAlign": "center"},
                ),
            ],
            className="corr-container",
        ),
        # Div 2 Heritage Dataframe vs HDI 2020
        html.Div(
            children=[
                html.H2("Heritage Dataframe vs HDI 2020:", className="title-graphs"),
                html.Hr(className="linha-separadora"),
                html.P("Correlation between Rankings:"),
                # Tabela 1
                dash_table.DataTable(
                    columns=[
                        {"name": "", "id": "index"},
                        {"name": "Ranking HDI 2020", "id": "Ranking HDI 2020"},
                        {
                            "name": "Ranking Economic Freedom",
                            "id": "Ranking Economic Freedom",
                        },
                    ],
                    data=[
                        {
                            "index": "Ranking HDI 2020",
                            "Ranking HDI 2020": correlation_rankings_heritage.iloc[
                                0, 0
                            ],
                            "Ranking Economic Freedom": correlation_rankings_heritage.iloc[
                                0, 1
                            ],
                        },
                        {
                            "index": "Ranking Economic Freedom",
                            "Ranking HDI 2020": correlation_rankings_heritage.iloc[
                                1, 0
                            ],
                            "Ranking Economic Freedom": correlation_rankings_heritage.iloc[
                                1, 1
                            ],
                        },
                    ],
                    style_table={
                        "maxWidth": "80%",
                        "margin": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"textAlign": "center"},
                ),
                html.P("Correlation between HDI 2020 and Economic Freedom:"),
                # Tabela 2
                dash_table.DataTable(
                    columns=[
                        {"name": "", "id": "index"},
                        {"name": "hdi_2020", "id": "hdi_2020"},
                        {
                            "name": "Economic Freedom Summary Index",
                            "id": "Economic Freedom Summary Index",
                        },
                    ],
                    data=[
                        {
                            "index": "hdi_2020",
                            "hdi_2020": correlation_hdi_ef_heritage.iloc[0, 0],
                            "Economic Freedom Summary Index": correlation_hdi_ef_heritage.iloc[
                                0, 1
                            ],
                        },
                        {
                            "index": "Economic Freedom Summary Index",
                            "hdi_2020": correlation_hdi_ef_heritage.iloc[1, 0],
                            "Economic Freedom Summary Index": correlation_hdi_ef_heritage.iloc[
                                1, 1
                            ],
                        },
                    ],
                    style_table={
                        "maxWidth": "80%",
                        "margin": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"textAlign": "center"},
                ),
            ],
            className="corr-container",
        ),
        # Div 3 Fraser Dataframe vs Heritage Dataframe
        html.Div(
            children=[
                html.H2(
                    "Fraser Dataframe vs Heritage Dataframe:", className="title-graphs"
                ),
                html.Hr(className="linha-separadora"),
                html.P("Correlation between Rankings:"),
                # Tabela 1
                dash_table.DataTable(
                    columns=[
                        {"name": "", "id": "index"},
                        {
                            "name": "Ranking Economic Freedom_x",
                            "id": "Ranking Economic Freedom_x",
                        },
                        {
                            "name": "Ranking Economic Freedom_y",
                            "id": "Ranking Economic Freedom_y",
                        },
                    ],
                    data=[
                        {
                            "index": "Ranking Economic Freedom_x",
                            "Ranking Economic Freedom_x": correlation_rankings_fraser_heritage.iloc[
                                0, 0
                            ],
                            "Ranking Economic Freedom_y": correlation_rankings_fraser_heritage.iloc[
                                0, 1
                            ],
                        },
                        {
                            "index": "Ranking Economic Freedom_y",
                            "Ranking Economic Freedom_x": correlation_rankings_fraser_heritage.iloc[
                                1, 0
                            ],
                            "Ranking Economic Freedom_y": correlation_rankings_fraser_heritage.iloc[
                                1, 1
                            ],
                        },
                    ],
                    style_table={
                        "maxWidth": "80%",
                        "margin": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"textAlign": "center"},
                ),
                html.P(
                    "Correlation between Economic Freedom Summary Index_x and Index_y:"
                ),
                # Tabela 2
                dash_table.DataTable(
                    columns=[
                        {"name": "", "id": "index"},
                        {
                            "name": "Economic Freedom Summary Index_x",
                            "id": "Economic Freedom Summary Index_x",
                        },
                        {"name": "Index_y", "id": "Index_y"},
                    ],
                    data=[
                        {
                            "index": "Economic Freedom Summary Index_x",
                            "Economic Freedom Summary Index_x": correlation_index_fraser_heritage.iloc[
                                0, 0
                            ],
                            "Index_y": correlation_index_fraser_heritage.iloc[0, 1],
                        },
                        {
                            "index": "Index_y",
                            "Economic Freedom Summary Index_x": correlation_index_fraser_heritage.iloc[
                                1, 0
                            ],
                            "Index_y": correlation_index_fraser_heritage.iloc[1, 1],
                        },
                    ],
                    style_table={
                        "maxWidth": "80%",
                        "margin": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"textAlign": "center"},
                ),
            ],
            className="corr-container",
        ),
        # Div para o Box Plot
        html.Div(
            [
                html.H2(children="Box Plot", className="title-graphs"),
                html.Hr(className="linha-separadora"),
                dcc.Graph(
                    id="box-plot", className="graph-show", figure=criar_box_plot()
                ),
            ],
            className="box-plot-container",
        ),
        # Div para o primeiro gráfico
        html.Div(
            [
                html.H2(
                    children="Correlations Over the Years",
                    className="title-graphs",
                ),
                html.Hr(className="linha-separadora"),
                dcc.Graph(
                    id="correlation-graph",
                    className="graph-show",
                    figure={
                        "data": [
                            {
                                "x": years,
                                "y": correlations,
                                "type": "bar",
                                "name": "Correlation",
                            },
                        ],
                        "layout": {
                            "xaxis": {"title": "Year"},
                            "yaxis": {"title": "Correlation"},
                        },
                    },
                ),
            ]
        ),
        # Div para o segundo gráfico
        html.Div(
            [
                html.H2(
                    id="ranking-title",
                    children="Ranking de EFI e HDI",
                    className="title-graphs",
                ),
                html.Hr(className="linha-separadora"),
                # Gráfico atualizado dinamicamente com base na entrada do usuário
                dcc.Graph(
                    id="ranking-graph",
                    className="graph-show",
                ),
                # dropdown para selecionar o código do país
                dcc.Dropdown(
                    id="iso-dropdown",
                    options=iso_options,
                    value="VEN",
                    multi=False,
                ),
                # Botão de atualização
                html.Button(
                    "Atualizar",
                    id="update-button",
                ),
            ],
            className="ranking-container",
        ),
        # Div para o gráfico de regressão linear
        html.Div(
            [
                html.H2(id="regression-title", children="Regressão Linear"),
                html.Hr(className="linha-separadora"),
                dcc.Graph(
                    id="regression-graph",
                    className="graph-show",
                ),
            ],
            className="regression-container",
        ),
    ],
)


@app.callback(
    [Output("ranking-graph", "figure"), Output("ranking-title", "children")],
    [Input("update-button", "n_clicks")],
    [State("iso-dropdown", "value")],
)
def update_graph(n_clicks, iso_value):
    if iso_value is None or iso_value == "":
        return dash.no_update, dash.no_update

    filter = df["iso3"] == iso_value

    hdi_ranks = []
    ef_ranks = []
    years = list(range(1996, 2022))

    for year in years:
        hdi_col = f"hdi_{year}"
        ef_col = f"ef_{year}"
        hdi_rank_col = f"hdi_{year}_rank"
        ef_rank_col = f"ef_{year}_rank"

        df[hdi_rank_col] = df[hdi_col].rank(ascending=False)
        df[ef_rank_col] = df[ef_col].rank(ascending=False)

    df_filter = df[filter]
    ef_ranks = df_filter[
        ["ef_" + str(year) + "_rank" for year in years]
    ].values.flatten()
    hdi_ranks = df_filter[
        ["hdi_" + str(year) + "_rank" for year in years]
    ].values.flatten()

    # Retorna a figura e o título atualizados
    figure = {
        "data": [
            {"x": years, "y": ef_ranks, "type": "line", "name": "EFI Rank"},
            {"x": years, "y": hdi_ranks, "type": "line", "name": "HDI Rank"},
        ],
        "layout": {
            "xaxis": {"title": "Ano"},
            "yaxis": {
                "title": "Ranking",
                "range": [185, 0],
                "autorange": False,
                "autotick": False,
                "tick0": 0,
                "dtick": 10,
                "showticklabels": True,
            },
            "shapes": [
                # Linha para o Quadrante 1
                {
                    "type": "line",
                    "x0": min(years),
                    "x1": max(years),
                    "y0": 46,
                    "y1": 46,
                    "line": {"color": "green", "width": 2, "dash": "dash"},
                },
                # Linha para o Quadrante 2
                {
                    "type": "line",
                    "x0": min(years),
                    "x1": max(years),
                    "y0": 92,
                    "y1": 92,
                    "line": {"color": "#F4CD01", "width": 2, "dash": "dash"},
                },
                # Linha para o Quadrante 3
                {
                    "type": "line",
                    "x0": min(years),
                    "x1": max(years),
                    "y0": 138,
                    "y1": 138,
                    "line": {"color": "#eb7575", "width": 2, "dash": "dash"},
                },
                # Linha para o Quadrante 4
                {
                    "type": "line",
                    "x0": min(years),
                    "x1": max(years),
                    "y0": 184,
                    "y1": 184,
                    "line": {"color": "red", "width": 2, "dash": "dash"},
                },
            ],
            "annotations": [
                {
                    "x": min(years) + 2,
                    "y": 46 - 5,
                    "xref": "x",
                    "yref": "y",
                    "text": "Quadrante 1",
                    "showarrow": False,
                    "font": {"color": "green"},
                },
                {
                    "x": min(years) + 2,
                    "y": 92 - 5,
                    "xref": "x",
                    "yref": "y",
                    "text": "Quadrante 2",
                    "showarrow": False,
                    "font": {"color": "#F4CD01"},
                },
                {
                    "x": min(years) + 2,
                    "y": 138 - 5,
                    "xref": "x",
                    "yref": "y",
                    "text": "Quadrante 3",
                    "showarrow": False,
                    "font": {"color": "#eb7575"},
                },
                {
                    "x": min(years) + 2,
                    "y": 184 - 5,
                    "xref": "x",
                    "yref": "y",
                    "text": "Quadrante 4",
                    "showarrow": False,
                    "font": {"color": "red"},
                },
            ],
        },
    }

    title = f'Ranking de EFI e HDI no(a) {df_filter["country"].values[0]} (1996-2021)'

    return figure, title


@app.callback(
    [Output("regression-graph", "figure"), Output("regression-title", "children")],
    [Input("update-button", "n_clicks")],
    [State("iso-dropdown", "value")],
)
def update_regression_graph(n_clicks, iso_value):
    if not n_clicks:
        return dash.no_update, dash.no_update

    if iso_value is None or iso_value == "":
        return dash.no_update, "Dados não disponíveis para o país selecionado."

    df_filter = df[df["iso3"] == iso_value]

    if df_filter.empty:
        return dash.no_update, "Dados não disponíveis para o país selecionado."

    # Verificar se há dados ausentes
    if df_filter[[f"ef_{year}_rank" for year in range(1996, 2022)]].isnull().any().any():
        return dash.no_update, "Dados ausentes para realizar a regressão."

    ef_ranks = df_filter[[f"ef_{year}_rank" for year in range(1996, 2022)]].values.flatten()
    hdi_ranks = df_filter[[f"hdi_{year}_rank" for year in range(1996, 2022)]].values.flatten()

    if np.isnan(ef_ranks).any() or np.isnan(hdi_ranks).any():
        return dash.no_update, "Dados contêm valores nulos (NaN). Não é possível realizar a regressão."

    reg_ef = LinearRegression().fit(
        np.array([[year] for year in range(1996, 2022)]), ef_ranks
    )

    reg_hdi = LinearRegression().fit(
        np.array([[year] for year in range(1996, 2022)]), hdi_ranks
    )


    years_pred = np.array([[year] for year in range(2022, 2030)])
    ef_rank_pred = reg_ef.predict(years_pred)
    hdi_rank_pred = reg_hdi.predict(years_pred)

    figure = {
        "data": [
            {
                "x": list(range(1996, 2022)),
                "y": ef_ranks,
                "type": "scatter",
                "mode": "markers",
                "name": "EF Rank (1996-2021)",
            },
            {
                "x": list(range(1996, 2022)),
                "y": hdi_ranks,
                "type": "scatter",
                "mode": "markers",
                "name": "HDI Rank (1996-2021)",
            },
            {
                "x": list(range(2022, 2030)),
                "y": ef_rank_pred,
                "type": "scatter",
                "mode": "lines+markers",
                "name": "EF Rank Predito (2022-2029)",
            },
            {
                "x": list(range(2022, 2030)),
                "y": hdi_rank_pred,
                "type": "scatter",
                "mode": "lines+markers",
                "name": "HDI Rank Predito (2022-2029)",
            },
        ],
        "layout": {
            "xaxis": {"title": "Ano"},
            "yaxis": {
                "title": "Ranking",
                "range": [185, 0],
                "autorange": False,
                "autotick": False,
                "tick0": 0,
                "dtick": 10,
                "showticklabels": True,
            },
            "title": f'Regressão Linear de EF e HDI no(a) {df_filter["country"].values[0]} (1996-2029)',
        },
    }

    title = f'Regressão Linear de EF e HDI no(a) {df_filter["country"].values[0]} (1996-2029)'

    return figure, title


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True)
    # app.run_server(debug=True, use_reloader=True, host='0.0.0.0', port=8080) # para rodar externo
