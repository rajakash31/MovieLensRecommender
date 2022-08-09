import statistics as st
import plotly.plotly as py
from plotly.graph_objs import *


def plot_eigenvalues(exp_variance, cum_variance, no_prin_comp):
    no_components = len(exp_variance)
    # Create a color list to segregate seed from chaff
    color_list = ['#447adb'] * no_components
    for i in range(no_prin_comp, no_components):
        color_list[i] = '#db5a44'

    # Create the traces for exp_variance and another for cum_variance
    trace_exp_var = Bar(
        x=[i for i in range(no_components)],
        y=exp_variance,
        name='Explained Variance',
        marker=Marker(color=color_list))

    trace_cum_var = Scatter(
        x=[i for i in range(no_components)],
        y=cum_variance,
        name='Cumulative Explained Variance',
        mode='markers',
        marker=Marker(color=color_list))

    # Create their corresponding graph layouts
    layout_exp_var = Layout(
        xaxis=XAxis(title='Principal Components'),
        yaxis=YAxis(title='Explained Variance in %'),
        title='Explained Variance by Components',
        autosize=True)
    layout_cum_var = Layout(
        xaxis=XAxis(title='Principal Components'),
        yaxis=YAxis(title='Explained Variance in %'),
        title='Cumulative Explained Variance by Components',
        autosize=True)

    # Create the data from the traces
    data_exp_var = Data([trace_exp_var])
    data_cum_var = Data([trace_exp_var, trace_cum_var])

    # Plot the graph with the data and the layout, also save an image
    fig_exp_var = Figure(data=data_exp_var, layout=layout_exp_var)
    fig_cum_var = Figure(data=data_cum_var, layout=layout_cum_var)

    py.image.save_as(fig_exp_var, 'exp_var_graph.png')
    py.image.save_as(fig_cum_var, 'cum_exp_var_graph.png')
    py.plot(fig_exp_var, filename=' Explained Variance by Component')
    py.plot(fig_cum_var, filename=' Cumulative Explained Variance by Component')


def plot_data_analysis(data):
    users_per_movie = []
    movies_per_user = []
    '''
    Find number of users rating a movie for all movies,
    then the mean and the median.
    '''
    for i in range(data.shape[1]):
        users_per_movie.append(data[:, i][data[:, i] != 0].shape[0])

    avg_users_per_movie = st.mean(users_per_movie)
    median_movie = st.median(users_per_movie)

    '''
    Find number of movies rated by each user,
    then the mean and the median.
    '''
    for i in range(data.shape[0]):
        movies_per_user.append(data[i][data[i] != 0].shape[0])

    avg_movies_per_user = st.mean(movies_per_user)
    median_user = st.median(movies_per_user)

    trace_users_per_movie = Bar(
        x=[i for i in range(data.shape[1])],
        y=users_per_movie,
        name='Users Per Movie'
        )
    trace_avg_movie = Scatter(
        x=[i for i in range(data.shape[1])],
        y=[avg_users_per_movie]*data.shape[1],
        name='Average number of ratings',
        line=Line(
            color='rgb(252,174,145)',
            opacity=1,
            width=1.5
            )
        )
    trace_median_movie = Scatter(
        x=[i for i in range(data.shape[1])],
        y=[median_movie]*data.shape[1],
        name='Median of ratings',
        line=Line(
            color='rgb(222,45,38)',
            opacity=1,
            width=1.2
            )
        )
    trace_movies_per_user = Bar(
        x=[i for i in range(data.shape[0])],
        y=movies_per_user,
        name='Movies Per User'
        )
    trace_avg_user = Scatter(
        x=[i for i in range(data.shape[0])],
        y=[avg_movies_per_user]*data.shape[0],
        name='Average number of movies',
        line=Line(
            color='rgb(252,174,145)',
            opacity=1,
            width=1.5
            )
        )
    trace_median_user = Scatter(
        x=[i for i in range(data.shape[0])],
        y=[median_user]*data.shape[0],
        name='Median of movies',
        line=Line(
            color='#rgb(222,45,38)',
            opacity=1,
            width=1.2
            )
        )

    layout_users_per_movie = Layout(
        xaxis=XAxis(title='Movies'),
        yaxis=YAxis(title='Users'),
        title='Number Of User Ratings Per Movie',
        showlegend=True,
        autosize=True
        )

    layout_movies_per_user = Layout(
        xaxis=XAxis(title='Users'),
        yaxis=YAxis(title='Movies'),
        title='Number Of Movies Rated Per User',
        showlegend=True,
        autosize=True
        )

    data_users_per_movie = Data([trace_users_per_movie, trace_avg_movie, trace_median_movie])
    data_movies_per_user = Data([trace_movies_per_user, trace_avg_user, trace_median_user])

    fig_users_per_movie = Figure(data=data_users_per_movie, layout=layout_users_per_movie)
    fig_movies_per_user = Figure(data=data_movies_per_user, layout=layout_movies_per_user)

    py.image.save_as(fig_users_per_movie, 'users_per_movie.png')
    py.image.save_as(fig_movies_per_user, 'movies_per_user.png')
    py.plot(fig_users_per_movie, filename=' Number Of User Ratings Per Movie')
    py.plot(fig_movies_per_user, filename=' Number Of Movies Rated Per User')
