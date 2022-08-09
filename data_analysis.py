# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
import auxiliary


def initialize_user_rating_matrix(n_users, n_movies):
    '''
    Creates a (n_users+1) x (n_movies+1) sized matrix.
    All entries in this matrix are zeros
    '''
    user_rating_matrix = np.zeros((n_users+1, n_movies+1), dtype=float)
    return user_rating_matrix


def parse_dataset(filename, user_rating_matrix):
    '''
    Reads a ratings data file and populates the user_rating_matrix.

    The first row of the user_ratings_matrix contains
    the average rating per movie from all users who have rated that movie

    The first column of the user_ratings_matrix contains contains
    the average rating given by each user for all movies rated by him.
    '''
    f = open(filename, 'r')
    for data in f:
        (user_id, item_id, rating, timestamp) = (data.strip().split())
        user_id = int(user_id)
        item_id = int(item_id)
        rating = float(rating)
        user_rating_matrix[user_id][item_id] = rating
    f.close()
    print("File closed after data read successfully.")

    print("Calculating averages...")
    # Average rating per movie from all users who rated it
    for item_id in range(1, user_rating_matrix.shape[1]):
        ratingNum = 0
        ratingSum = 0.0
        for user_id in range(1, user_rating_matrix.shape[0]):
            if (user_rating_matrix[user_id][item_id] != 0):
                        ratingNum += 1
                        ratingSum += user_rating_matrix[user_id][item_id]

        user_rating_matrix[0][item_id] = ratingSum/ratingNum

    # Average rating per uesr for all movies rated by him
    for user_id in range(1, user_rating_matrix.shape[0]):
        ratingNum = 0
        ratingSum = 0.0
        for item_id in range(1, user_rating_matrix.shape[1]):
            if (user_rating_matrix[user_id][item_id] != 0):
                ratingNum += 1
                ratingSum += user_rating_matrix[user_id][item_id]

        user_rating_matrix[user_id][0] = ratingSum/ratingNum

    print("Average values entered.")


def replace_zero_with_mean(ratings_matrix):
    '''
    This method implements mean value substitution
    for missing values.
    Replaces all zeros in the ratings_matrix
    with the mean value for that corresponding row,
    that is for every user, his own mean.
    '''
    for row in range(0, ratings_matrix.shape[0]):
        ratings_matrix[row][ratings_matrix[row] == 0] = ratings_matrix[row][0]
    return ratings_matrix


def standardize_data(data):
    '''
    Returns standardized data, that is,
    the transformed data has zero mean and
    unit standard deviation
    '''
    return StandardScaler().fit_transform(data)


def eigenanalysis(corr_matrix):
    '''
    Performs eigenvalueanalysis of the correlation matrix,
    sorts the results by eigen value and
    returns a list of explained variance per component,
    and a list of cumulative explained variance per component,
    both in percentage.
    '''
    eig_vals, eig_vecs = np.linalg.eig(corr_matrix)
    no_components = len(eig_vals)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(no_components)]
    # Sort the (eigenvalue, eigenvector) tuples from high to low on eigenvalue
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True)
    sorted_eig_vals = [eig_pairs[i][0] for i in range(no_components)]

    tot = sum(sorted_eig_vals)
    # Find explained variance per component, also find cumulative explained variance
    exp_var = [(val / tot)*100 for val in sorted_eig_vals]
    cum_exp_var = np.cumsum(exp_var)

    return exp_var, cum_exp_var


def get_principal_components(cum_var, energy_threshold):
    '''
    From a list of cumulative explained variance,
    it returns the number of components that, together,
    contribute at least energy_threshold information
    '''
    count_prin_comp = 0
    for val in cum_var:
        if val <= 90.0:
            count_prin_comp += 1
        else:
            break
    count_prin_comp += 1
    print("Principal Components explaining %s%% variance: %s" % (energy_threshold, count_prin_comp))
    return count_prin_comp


def pca(dataset, no_components):
    '''
    Performs PCA on the dataset, retaining
    no_components principal no_components
    '''
    pca_space = sklearnPCA(n_components=no_components)
    return pca_space.fit_transform(data_std)


print("Initializing matrix....")
user_rating_matrix = initialize_user_rating_matrix(943, 1682)
print("Parsing data....")
parse_dataset('D:\\Documents\\College\\8th Sem\\MovieLens\\ml-100k\\u.data', user_rating_matrix)
print("Populating User-Ratings matrix is done.")
joblib.dump(user_rating_matrix, 'Pickled\\user_rating_matrix.pkl')
print("User-Ratings matrix is serialized to disk.")

data = user_rating_matrix[1:, 1:]
data_std = StandardScaler().fit_transform(data)

# Plot insights from data
auxiliary.plot_data_analysis(data)

data_corr_mat = np.corrcoef(data.T)
joblib.dump(data_corr_mat, 'Pickled\\data_corr_mat.pkl')
print("Correlation matrix is serialized to disk.")
exp_var, cum_var = eigenanalysis(data_corr_mat)
no_comps = get_principal_components(cum_var, 90)

auxiliary.plot_eigenvalues(exp_var, cum_var, no_comps)

data_pca = pca(data_std, no_comps)
joblib.dump(data_pca, 'Pickled\\data_pca_reduced.pkl')
print("Dimensionally-reduced data is serialized to disk.")
print(data_pca.shape)
