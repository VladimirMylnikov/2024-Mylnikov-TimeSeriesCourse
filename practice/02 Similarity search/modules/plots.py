import numpy as np
import pandas as pd

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot

def plot_ts_set(ts_set: np.ndarray, title: str = 'Input Time Series Set', filename='ts_set') -> None:
    """
    Plot the time series set

    Parameters
    ----------
    ts_set: time series set
    title: title of plot
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title=title,
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black'))
                      )

    plot(fig, filename=f'{filename}.html')


def mplot2d(x: np.ndarray, y: np.ndarray, plot_title: str = None, x_title: str = None, y_title: str = None, trace_titles: np.ndarray = None, filename='mplot2d') -> None:
    """
    Multiple 2D Plots on figure for different experiments

    Parameters
    ----------
    x: values of x axis of plot
    y: values of y axis of plot
    plot_title: title of plot
    x_title: title of x axis of plot
    y_title: title of y axis of plot
    trace_titles: titles of plot traces (lines)
    """

    fig = go.Figure()

    for i in range(y.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=y[i], line=dict(width=3), name=trace_titles[i]))

    fig.update_xaxes(showgrid=False,
                     title=x_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2,
                     tickvals=x)
    fig.update_yaxes(showgrid=False,
                     title=y_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black')),
                      width=1000,
                      height=600
                      )

    plot(fig, filename=f'{filename}.html')


def plot_bestmatch_data(ts: np.ndarray, query: np.ndarray, filename='bestmatch') -> None:
    """
    Visualize the input data (time series and query) for the best match task

    Parameters
    ----------
    ts: time series
    query: query
    """

    query_len = query.shape[0]
    ts_len = ts.shape[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.1, 0.9], subplot_titles=("Query", "Time Series"), horizontal_spacing=0.04)

    fig.add_trace(go.Scatter(x=np.arange(query_len), y=query, line=dict(color=px.colors.qualitative.Plotly[1])),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color=px.colors.qualitative.Plotly[0])),
                row=1, col=2)

    fig.update_annotations(font=dict(size=24, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      title_x=0.5)

    plot(fig, filename=f'{filename}.html')


def plot_bestmatch_results(ts: np.ndarray, query: np.ndarray, bestmatch_results: dict) -> None:
    """
    Visualize the best match results

    Parameters
    ----------
    ts: time series
    query: query
    bestmatch_results: output data found by the best match algorithm
    """

    # Создаем график
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Рисуем временной ряд
    ax.plot(ts, label='Time Series')
    
    # Выделяем образец поиска
    # query_indices = np.arange(bestmatch_results['indices'][0], bestmatch_results['indices'][0] + len(query))
    # ax.plot(query_indices, ts[query_indices], label='Query', color='red')
    
    # Выделяем найденные подпоследовательности
    for idx in bestmatch_results['indices']:
        subseq_indices = np.arange(idx, idx + len(query))
        ax.plot(subseq_indices, ts[subseq_indices], label=f'Match {bestmatch_results["indices"].index(idx)}', color='green')
    
    # Добавляем подписи и легенду
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()


def pie_chart(labels: np.ndarray, values: np.ndarray, plot_title='Pie chart', filename="pie_chart") -> None:
    """
    Build the pie chart

    Parameters
    ----------
    labels: sector labels
    values: values
    """

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_traces(textfont_size=20)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      legend=dict(font=dict(size=20, color='black')),
                      width=700,
                      height=500
                      )
    # iplot(fig)
    # fig.show(renderer="browser")

    plot(fig, filename=f'{filename}.html')