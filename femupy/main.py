import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from scipy.stats.qmc import Halton, LatinHypercube
import matplotlib.colorbar as cbar
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from matplotlib.patches import Patch

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

GnBu_colors = [
    "#f7fcf0", "#e0f3db", "#ccebc5", "#a8ddb5", "#7bccc4",
    "#4eb3d3", "#2b8cbe", "#0868ac", "#084081"]

def array_plot(array, label_1=None, label_2=None, ticklabels_1=None, ticklabels_2=None, display_plot=True, **kwargs):
    '''possible kwargs: title, cbar_label, xtick_rotation, ytick_rotation'''
    dim_y, dim_x = array.shape
    fig, ax = plt.subplots()
    cax = ax.pcolormesh(array, cmap='GnBu', edgecolors='white', linewidth=1, vmin=0, vmax=1)
    texts = []
    for i in range(dim_y):
        row_texts = []
        for j in range(dim_x):
            value = array[i, j]
            color = 'white' if round(100*value) >= 70 else 'black'
            text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', color=color, fontsize=10)
            row_texts.append(text)
        texts.append(row_texts)
    cb = fig.colorbar(cax, ax=ax, shrink=0.4)
    cb.set_ticks([0,  1])
    if 'cbar_label' in kwargs.keys():
        cb.set_label(kwargs['cbar_label'],labelpad=-5)
    ax.set_xticks(np.arange(dim_x) + 0.5)
    ax.set_yticks(np.arange(dim_y) + 0.5)
    if ticklabels_1 is not None:
        if isinstance(ticklabels_1[0],(float,int)):
            ticklabels_1 = list(map("{:.2f}".format,ticklabels_1))
    else:
        ticklabels_1 = np.array(range(1,dim_y+1),dtype=str)
    yrot = kwargs['ytick_rotation'] if 'ytick_rotation' in kwargs.keys() else 0
    ax.set_yticklabels(ticklabels_1,rotation = yrot,fontsize=10)
    if ticklabels_2 is not None:
        if isinstance(ticklabels_2[0],(float,int)):
            ticklabels_2 = list(map("{:.2f}".format,ticklabels_2))
    else:
        ticklabels_2 = np.array(range(1,dim_x+1),dtype=str)
    xrot = kwargs['xtick_rotation'] if 'xtick_rotation' in kwargs.keys() else 0
    ax.set_xticklabels(ticklabels_2,rotation = xrot,fontsize=10)
    labels_y = label_1
    labels_x = label_2
    ax.set_xlabel(labels_x, fontsize=10, labelpad=10)
    ax.set_ylabel(labels_y, fontsize=10, labelpad=10)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    ax.tick_params(axis='both', which='both', length=0)
    fig.set_facecolor('#e6e6e6')
    ax.set_aspect('equal', adjustable='box')
    if display_plot:
        plt.show()
    return fig, ax

# def FMAC_plot(mac_matrix, eigenfrequencies_model, eigenfrequencies_exp, show_offdiagonal=False, ylim=30, **kwargs):
    def rgb_to_hex(rgb_tuple):
        r,g,b = rgb_tuple
        r,g,b = int(r),int(g),int(b)
        return f'#{r:02x}{g:02x}{b:02x}'

    eigenfrequencies_exp_extended = np.repeat(eigenfrequencies_exp[:,np.newaxis],eigenfrequencies_model.shape[0],axis=1)
    eigenfrequencies_model_extended = np.repeat(eigenfrequencies_model[np.newaxis,:],eigenfrequencies_exp.shape[0],axis=0)
    frequency_ratio_matrix = (eigenfrequencies_model_extended/eigenfrequencies_exp_extended)
    X_modes = np.repeat(np.arange(len(eigenfrequencies_exp))[:,np.newaxis],eigenfrequencies_model.shape[0],axis=1)+1
    Y_frequencies = (frequency_ratio_matrix-1)*100
    R_macs = mac_matrix.copy()
    colors = np.zeros_like(mac_matrix,dtype='<U7')
    rgb_array = 255*plt.cm.GnBu(mac_matrix)[:,:,:-1]
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            colors[i,j] = rgb_to_hex(rgb_array[i,j])
    fig,ax = plt.subplots(1,figsize=(6,5))
    ax.grid(axis='y', zorder=-2)
    ax.axhline(0,0,1,c='k',linewidth=1, dashes=(100,0.005), zorder=1)
    ax.set_xlabel('Experimental mode')
    ax.set_ylabel('Relative error of FEM frequency [%]')
    ax.set_xticks(np.arange(1,len(Y_frequencies)+1))
    ax.set_axisbelow(True)
    facecolor = kwargs['facecolor'] if 'facecolor' in kwargs.keys() else '#E6E6E6'
    ax.set_facecolor(facecolor)
    if not show_offdiagonal:
        conf_matrix = confidence_matrix(mac_matrix)
        _, order_1, order_2 = sort_by_diagonal(conf_matrix)
        indices = [(i,j) for i,j in zip(order_1,order_2)]
        rows, cols = zip(*indices)
        flat_indices = np.ravel_multi_index((rows, cols), X_modes.shape)
        X_modes = np.take(X_modes, flat_indices)
        Y_frequencies = np.take(Y_frequencies, flat_indices)
        R_macs = np.take(R_macs, flat_indices)
        colors = np.take(colors, flat_indices)
    else:
        X_modes = X_modes.flatten()
        Y_frequencies = Y_frequencies.flatten()
        R_macs = R_macs.flatten()
        colors = colors.flatten()
    sc = plt.scatter(X_modes, Y_frequencies, marker='o',s=200*R_macs, edgecolors='k',linewidths=1,alpha=1,facecolor=colors,zorder=2)
    plt.ylim((-ylim,ylim))
    cax, _ = cbar.make_axes(ax,shrink=0.4,aspect=30) 
    cb = cbar.ColorbarBase(cax, cmap='GnBu')
    cb.set_ticks([0,  1])
    cb.ax.set_ylabel('MAC value', labelpad=-5)
    return fig, ax, sc

# def FMAC_plot(mac_matrix, eigenfrequencies_model, eigenfrequencies_exp, show_offdiagonal=False, ylim=30, **kwargs):
    X_modes, Y_frequencies, R_macs, colors = prepare_FMAC_input(mac_matrix, eigenfrequencies_model, eigenfrequencies_exp, show_offdiagonal)
    fig,ax = plt.subplots(1,figsize=(6,5))
    ax.grid(axis='y', zorder=-2)
    ax.axhline(0,0,1,c='k',linewidth=1, dashes=(100,0.005), zorder=1)
    ax.set_xlabel('Experimental mode')
    ax.set_ylabel('Relative error of FEM frequency [%]')
    ax.set_xticks(np.arange(1,len(Y_frequencies)+1))
    ax.set_axisbelow(True)
    facecolor = kwargs['facecolor'] if 'facecolor' in kwargs.keys() else '#E6E6E6'
    ax.set_facecolor(facecolor)
    sc = plt.scatter(X_modes, Y_frequencies, marker='o',s=R_macs, edgecolors='k',linewidths=1,alpha=1,facecolor=colors,zorder=2)
    plt.ylim((-ylim,ylim))
    cax, _ = cbar.make_axes(ax,shrink=0.4,aspect=30) 
    cb = cbar.ColorbarBase(cax, cmap='GnBu')
    cb.set_ticks([0,  1])
    cb.ax.set_ylabel('MAC value', labelpad=-5)
    return fig, ax, sc

def prepare_FMAC_input(mac_matrix, frequencies_model, frequencies_measurement, show_offdiagonal=False):
    def rgb_to_hex(rgb_tuple):
        r,g,b = rgb_tuple
        r,g,b = int(r),int(g),int(b)
        return f'#{r:02x}{g:02x}{b:02x}'
    frequencies_measurement_extended = np.repeat(frequencies_measurement[:,np.newaxis],frequencies_model.shape[0],axis=1)
    frequencies_model_extended = np.repeat(frequencies_model[np.newaxis,:],frequencies_measurement.shape[0],axis=0)
    frequency_ratio_matrix = (frequencies_model_extended/frequencies_measurement_extended)
    X_modes = np.repeat(np.arange(len(frequencies_measurement))[:,np.newaxis],frequencies_model.shape[0],axis=1)+1
    Y_frequencies = (frequency_ratio_matrix-1)*100
    R_macs = mac_matrix.copy()*200
    colors = np.zeros_like(mac_matrix,dtype='<U7')
    rgb_array = 255*plt.cm.GnBu(mac_matrix)[:,:,:-1]
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            colors[i,j] = rgb_to_hex(rgb_array[i,j])
    if not show_offdiagonal:
        conf_matrix = confidence_matrix(mac_matrix)
        _, order_1, order_2 = sort_by_diagonal(conf_matrix)
        indices = [(i,j) for i,j in zip(order_1,order_2)]
        rows, cols = zip(*indices)
        flat_indices = np.ravel_multi_index((rows, cols), X_modes.shape)
        X_modes = np.take(X_modes, flat_indices)
        Y_frequencies = np.take(Y_frequencies, flat_indices)
        R_macs = np.take(R_macs, flat_indices)
        colors = np.take(colors, flat_indices)
    else:
        X_modes = X_modes.flatten()
        Y_frequencies = Y_frequencies.flatten()
        R_macs = R_macs.flatten()
        colors = colors.flatten()
    return X_modes, Y_frequencies, R_macs, colors

def MAC_matrix(eigenvectors_1, eigenvectors_2):
    '''Inputs eigenvectors_1 and eigenvectors_2 are of type np.ndarray.
    The first axis is the number of eigenvectors, and the second is number of components.'''
    dim_1 = eigenvectors_1.shape[0]
    dim_2 = eigenvectors_2.shape[0]
    D = np.dot(eigenvectors_1,eigenvectors_2.T)
    D1 = np.dot(np.conjugate(eigenvectors_1),eigenvectors_1.T)
    D2 = np.dot(np.conjugate(eigenvectors_2),eigenvectors_2.T)
    A = np.tile(np.diag(D1)[:,np.newaxis],dim_2)
    B = np.tile(np.diag(D2)[:,np.newaxis],dim_1).T
    MAC = D**2/(A*B)
    return MAC

def diag_MAC_matrix(eigenvectors_1, eigenvectors_2):
    '''Compute the diagonal terms of the MAC matrix.
    Input parameters eigenvector_1 and eigenvectors_2 should be of the same size'''
    assert eigenvectors_1.shape == eigenvectors_2.shape, "Eigenvectors must be of the same size"
    D = np.sum(eigenvectors_1 * np.conjugate(eigenvectors_2), axis=1)
    D1 = np.sum(eigenvectors_1 * np.conjugate(eigenvectors_1), axis=1)
    D2 = np.sum(eigenvectors_2 * np.conjugate(eigenvectors_2), axis=1)
    MAC_diag = (D**2)/(D1*D2)
    return MAC_diag

def normalize_eigenvectors(input_eigenvectors):
    if input_eigenvectors.ndim==2:
        norms = np.linalg.norm(input_eigenvectors, axis=1, keepdims=True)
    elif input_eigenvectors.ndim==3:
        norms = np.linalg.norm(input_eigenvectors, axis=2, keepdims=True)
    else:
        raise KeyError(f'The input has unexpected dimension of {input_eigenvectors.ndim}.')
    normalized_eigenvectors = input_eigenvectors / norms
    return normalized_eigenvectors

def compute_dot_products(input_eigenvectors, reference_eigenvectors):
    num_of_samples = input_eigenvectors.shape[0]
    num_of_modes = input_eigenvectors.shape[1]
    num_of_components = input_eigenvectors.shape[2]
    input_eigenvectors_reshaped = input_eigenvectors.reshape(-1, num_of_components)
    dot_products = np.dot(input_eigenvectors_reshaped, reference_eigenvectors.T)
    dot_products = dot_products.reshape(num_of_samples, num_of_modes, -1)
    return dot_products

def normalize_and_flip_eigenvectors(input_eigenvectors, reference_eigenvectors=None):
    input_eigenvectors = normalize_eigenvectors(input_eigenvectors)
    reference_eigenvectors = input_eigenvectors[0].copy() if reference_eigenvectors is None else reference_eigenvectors
    dot_products = compute_dot_products(input_eigenvectors, reference_eigenvectors)
    max_abs_indices = np.argmax(np.abs(dot_products), axis=-1)
    max_abs_signs = np.sign(dot_products[np.arange(dot_products.shape[0])[:, None], np.arange(dot_products.shape[1]), max_abs_indices])
    flipped_eigenvectors = input_eigenvectors*max_abs_signs[:,:,np.newaxis]
    return flipped_eigenvectors

def sort_by_diagonal(array):
    sorted_array = array.copy()
    num_rows, num_cols = sorted_array.shape
    row_order = np.arange(num_rows)
    col_order = np.arange(num_cols)
    dim = min(num_rows, num_cols)
    for i in range(dim):
        subarray = sorted_array[i:, i:]
        max_index = np.unravel_index(np.argmax(subarray, axis=None), subarray.shape)
        max_index = (max_index[0] + i, max_index[1] + i)
        if i!=max_index[0]:
            sorted_array[[i, max_index[0]]] = sorted_array[[max_index[0], i]]
            row_order[[i, max_index[0]]] = row_order[[max_index[0], i]]
        if i!=max_index[1]:
            sorted_array[:, [i, max_index[1]]] = sorted_array[:, [max_index[1], i]]
            col_order[[i, max_index[1]]] = col_order[[max_index[1], i]]
    return sorted_array, row_order, col_order

def confidence_matrix(mac_matrix):
    confidence_matrix = np.zeros_like(mac_matrix)
    num_rows, num_cols = confidence_matrix.shape
    for i in range(num_rows):
        for j in range(num_cols):
            confidence_matrix[i,j] = mac_matrix[i,j]-np.max(np.concatenate((np.delete(mac_matrix[i],j),np.delete(mac_matrix[:,j],i))))
    return confidence_matrix

def match_modes(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement): 
    mac_matrix = MAC_matrix(eigenvectors_measurement, eigenvectors_model)
    conf_matrix = confidence_matrix(mac_matrix)
    sorted_conf_matrix, order_meas, order_model = sort_by_diagonal(conf_matrix)
    confidences = np.diagonal(sorted_conf_matrix)
    indices = [(i,j) for i,j in zip(order_meas,order_model)]
    rows, cols = zip(*indices)
    len_meas = len(frequencies_measurement)
    len_model = len(frequencies_model)
    flat_indices = np.ravel_multi_index((rows, cols), (len_meas,len_model))
    min_len = min(len_meas,len_model)
    order_meas = order_meas[:min_len]
    order_model = order_model[:min_len]
    sort_ind = np.argsort(order_meas)
    order_meas = order_meas[sort_ind]
    order_model = order_model[sort_ind]
    confidences = confidences[sort_ind]
    frequencies_model = frequencies_model[order_model]
    frequencies_measurement = frequencies_measurement[order_meas]
    macs = mac_matrix.flatten()[flat_indices][sort_ind]
    return frequencies_model, frequencies_measurement, macs, order_model, order_meas, confidences

def compute_errors(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement):
    freqs_model, freqs_meas, macs, _, _, _ = match_modes(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement)
    frequency_errors = (freqs_model-freqs_meas)/freqs_meas
    mac_errors = 1-macs
    return frequency_errors, mac_errors

def compute_objective_function(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement, weights=None):
    errors_frequency, errors_mac = compute_errors(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement)
    if weights is None:
        weights = np.ones_like(errors_frequency)
    objective_frequency = np.sum(np.abs(errors_frequency)*weights)/np.sum(weights)
    objective_mac = np.sum(errors_mac*weights)/np.sum(weights)
    return objective_frequency, objective_mac

class SurrogateModel():

    def __init__(self, parameter_names, parameter_bounds, parameter_logarithmic=False, verbose=True):
        '''parameter_names: list of parameter names
        paramaeter_bounds: list of tuples with lower and upper bound
        logarithmic: boolean or list of booleans specifying if parameters should be normalised by a logarithmic scale'''
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.num_of_params = len(parameter_names)
        assert len(parameter_bounds)==self.num_of_params, 'Incompatible dimensions of parameter_names and parameter_bounds.'
        if type(parameter_logarithmic)==bool:
            self.parameter_logarithmic = [parameter_logarithmic for _ in range(self.num_of_params)]
        elif type(parameter_logarithmic)==list:
            assert len(parameter_logarithmic)==self.num_of_params, 'Incompatible dimensions of parameter_names and logarithmic.'
            assert all(isinstance(item, bool) for item in parameter_logarithmic)
            self.parameter_logarithmic = parameter_logarithmic
        if verbose:
            print(f'Number of parameters: {self.num_of_params}')
            for name, bounds, is_log in zip(parameter_names,parameter_bounds,self.parameter_logarithmic):
                log_lin = 'Log' if is_log else 'Linear'
                print(f'{name}: ({bounds[0]},{bounds[1]}), {log_lin}')

    def generate_samples(self, N, method='halton'):
        '''N: number of sample points to generate
        allowed methods: "halton", "lhs", and "uniform" '''
        dim = self.num_of_params
        if method == 'halton':
            sampler = Halton(d=dim, scramble=True)
            samples = sampler.random(n=N)*2 - 1
        elif method == 'lhs':
            sampler = LatinHypercube(d=dim)
            samples = sampler.random(n=N)*2 - 1
        elif method == 'uniform':
            samples = np.random.uniform(low=-1, high=1, size=(N, dim))
        else:
            raise ValueError("Method not recognized. Use 'halton', 'lhs', or 'uniform'.")
        samples = self._from_dimensionless(samples)
        return samples

    def default_parameters(self):
        return self._from_dimensionless(np.zeros(self.num_of_params))[0]

    def add_data(self, parameters, frequencies, eigenvectors):
        self.num_of_samples = parameters.shape[0]
        self.num_of_modes = frequencies.shape[1]
        self.num_of_nodes = eigenvectors.shape[2]
        assert frequencies.shape[0]==self.num_of_samples, 'Incompatible dimensions of parameters and frequencies.'
        assert eigenvectors.shape[0]==self.num_of_samples, 'Incompatible dimensions of parameters and eigenvectors.'
        assert eigenvectors.shape[1]==self.num_of_modes, 'Incompatible dimensions of frequencies and eigenvectors.'
        self.parameters = parameters
        self.frequencies = frequencies
        eigenvectors = normalize_and_flip_eigenvectors(eigenvectors)
        self.eigenvectors = eigenvectors

    def _to_dimensionless(self, parameters):
        assert parameters.ndim<=2, 'There are too many dimensions of the parameters input.'
        if parameters.ndim==1:
            parameters = parameters[np.newaxis,:].copy()
        else:
            parameters = parameters.copy()
        bounds = np.array(self.parameter_bounds)
        logarithmic = np.array(self.parameter_logarithmic)
        bounds[logarithmic] = np.log10(bounds[logarithmic])
        low_bounds = bounds[:,0]
        range_bounds = bounds[:,1]-bounds[:,0]
        parameters[:,logarithmic] = np.log10(parameters[:,logarithmic])
        parameters_dimensionless = 2*(parameters-low_bounds)/range_bounds-1
        return parameters_dimensionless
    
    def _from_dimensionless(self, parameters_dimensionless):
        assert parameters_dimensionless.ndim<=2, 'There are too many dimensions of the parameters input.'
        if parameters_dimensionless.ndim==1:
            parameters_dimensionless = parameters_dimensionless[np.newaxis,:].copy()
        else:
            parameters_dimensionless = parameters_dimensionless.copy()
        bounds = np.array(self.parameter_bounds)
        logarithmic = np.array(self.parameter_logarithmic)
        bounds[logarithmic] = np.log10(bounds[logarithmic])
        low_bounds = bounds[:,0]
        range_bounds = bounds[:,1]-bounds[:,0]
        parameters = (parameters_dimensionless+1)/2*range_bounds+low_bounds
        parameters[:,logarithmic] = 10**parameters[:,logarithmic]
        return parameters

    def train(self,method='ANN',**kwargs):
        '''For method = "ANN", kwargs are neurons_per_layer=(16,64,256), dropout=0, epoch=20, batch_size=8, learning_rate=0.001'''
        parameters_dimensionless = self._to_dimensionless(self.parameters)
        frequencies = self.frequencies
        eigenvectors = self.eigenvectors
        self.method = method
        if method=='ANN':
            for key in kwargs.keys():
                if key not in {'neurons_per_layer', 'dropout', 'epoch', 'batch_size', 'learning_rate'}:
                    raise KeyError(f"Unexpected argument: '{key}'' ")
            # model._train(parameters,frequencies,eigenvectors,**kwargs)
            model = SurrogateModel._train_ANN(parameters_dimensionless,frequencies,eigenvectors,**kwargs)
            self._model = model
    
    def predict(self,parameters,dofs=None):
        '''Predict frequencies and eigenvectors according to the trained model.
        dofs (np.ndarray): if only selected nodes from the eigenvectors are required, pass the indices of the correct nodes.'''
        parameters_dimensionless = self._to_dimensionless(parameters)
        frequencies, eigenvectors = self._predict_dimensionless(parameters_dimensionless,dofs)
        return frequencies, eigenvectors
    
    def _predict_dimensionless(self,parameters_dimensionless,dofs=None):
        '''Predict frequencies and eigenvectors according to the trained model using dimensionless parameter values.
        dofs (np.ndarray): if only selected nodes from the eigenvectors are required, pass the indices of the correct nodes.'''
        model = self._model
        if self.method=='ANN':
            frequencies, eigenvectors = SurrogateModel._predict_ANN(model, parameters_dimensionless)
        if dofs is not None:
            eigenvectors = eigenvectors[:,:,dofs]
        return frequencies, eigenvectors

    def cross_validate(self, fold=9, method='ANN', **kwargs):
        '''For method = "ANN", kwargs are neurons_per_layer=(16,64,256), dropout=0, epoch=20, batch_size=8, learning_rate=0.001'''
        kfold = KFold(n_splits=fold, shuffle=True)
        errors_frequencies = np.zeros((self.num_of_samples,self.num_of_modes))
        errors_eigenvectors = np.zeros((self.num_of_samples,self.num_of_modes))
        k = 0
        for train_index, test_index in kfold.split(self.parameters):
            parameters_train, parameters_test = self._to_dimensionless(self.parameters[train_index]), self._to_dimensionless(self.parameters[test_index])
            frequencies_train, frequencies_test = self.frequencies[train_index], self.frequencies[test_index]
            eigenvectors_train, eigenvectors_test = self.eigenvectors[train_index], self.eigenvectors[test_index]
            if method=='ANN':
                for key in kwargs.keys():
                    if key not in {'neurons_per_layer', 'dropout', 'epoch', 'batch_size', 'learning_rate'}:
                        raise KeyError(f"Unexpected argument: '{key}'' ")
                model = SurrogateModel._train_ANN(parameters_train,frequencies_train,eigenvectors_train,**kwargs)
                frequencies_predict, eigenvectors_predict = SurrogateModel._predict_ANN(model,parameters_test)
            errors_frequencies[test_index] = (frequencies_predict-frequencies_test)/frequencies_test
            errors_eigenvectors[test_index] = (1-diag_MAC_matrix(eigenvectors_predict.reshape(-1,self.num_of_nodes),eigenvectors_test.reshape(-1,self.num_of_nodes))).reshape(-1,self.num_of_modes)
            k+=1
            print(f"K-fold cross-validation: {k}/{fold} folds completed", end='\r')
        cross_validation = CrossValidation(self, errors_frequencies, errors_eigenvectors)
        return cross_validation
    
    def predict_default(self):
        parameters_dimensionless = np.zeros(self.num_of_params)
        frequencies, eigenvectors = self._predict_dimensionless(parameters_dimensionless)
        return frequencies, eigenvectors

    def compute_sensitivities(self, N=1024):
        problem = {'num_vars': self.num_of_params,
                'names': self.parameter_names,
                'bounds': np.array([(-1,1) for _ in self.parameter_names])}
        param_values = sample(problem, N)
        Y_freq = self._predict_dimensionless(param_values)[0]
        sensitivity_matrix_S1 = np.zeros((self.num_of_modes,self.num_of_params))
        sensitivity_matrix_ST = np.zeros((self.num_of_modes,self.num_of_params))
        for mode in range(self.num_of_modes):
            sensitivity_matrix_S1[mode] = analyze(problem, Y_freq[:,mode])['S1']
            sensitivity_matrix_ST[mode] = analyze(problem, Y_freq[:,mode])['ST']
        self.sensitivity_matrices = (sensitivity_matrix_S1, sensitivity_matrix_ST)

    def plot_sensitivities(self, metric='S1',display_plot=True):
        try:
            sensitivity_matrix_S1, sensitivity_matrix_ST = self.sensitivity_matrices
        except:
            raise ValueError('You need to first run compute_sensitivities function.')
        ticklabels_2 = self.parameter_names
        max_length = max([len(ticklabel) for ticklabel in ticklabels_2])
        if max_length < 8:
            xrot = 0
        elif max_length < 10:
            xrot = 30
        elif max_length < 12:
            xrot = 45
        elif max_length < 15:
            xrot = 60
        else:
            xrot = 90
        if metric=='S1':
            fig,ax = array_plot(sensitivity_matrix_S1,label_1='Mode', label_2='Parameter',ticklabels_2=ticklabels_2,title='Sensitivities', cbar_label='S1',xtick_rotation=xrot, display_plot=display_plot);
        elif metric=='ST':
            fig,ax = array_plot(sensitivity_matrix_ST,label_1='Mode', label_2='Parameter',ticklabels_2=ticklabels_2,title='Sensitivities', cbar_label='ST',xtick_rotation=xrot, display_plot=display_plot);
        else:
            raise ValueError('The metric should be either "S1" or "ST".')
        return fig,ax
        
    @staticmethod
    def _predict_ANN(model, parameters):
        assert parameters.ndim<=2, 'There are too many dimensions of the parameters input.'
        if parameters.ndim==1:
            X_predict = parameters[np.newaxis,:].copy()
        else:
            X_predict = parameters.copy()
        y_predict = model.predict(X_predict,verbose=False)
        frequencies = y_predict[:,:,0].astype(np.float64)
        eigenvectors = y_predict[:,:,1:].astype(np.float64)
        return frequencies, eigenvectors    
    
    def _train_ANN(parameters_train, frequencies_train, eigenvectors_train, neurons_per_layer=32, dropout=0.0, epoch=20, batch_size=8, learning_rate=0.001):
        num_of_samples = parameters_train.shape[0]
        num_of_params = parameters_train.shape[1]
        num_of_modes = frequencies_train.shape[1]
        num_of_nodes = eigenvectors_train.shape[2]
        assert frequencies_train.shape[0]==num_of_samples, 'Incompatible dimensions of parameters and frequencies.'
        assert eigenvectors_train.shape[0]==num_of_samples, 'Incompatible dimensions of parameters and eigenvectors.'
        assert eigenvectors_train.shape[1]==num_of_modes, 'Incompatible dimensions of frequencies and eigenvectors.'
        X_train = parameters_train.copy()
        y_train = np.concatenate((frequencies_train[:,:,np.newaxis],eigenvectors_train),axis=-1).copy()
        if isinstance(neurons_per_layer,int):
            num_of_layers = 1
            neurons_per_layer = [neurons_per_layer]
        elif isinstance(neurons_per_layer, (list, tuple, np.ndarray)):
            num_of_layers = len(neurons_per_layer)
        else:
            raise TypeError("Unsupported type for 'neurons_per_layer'.")
        input_neurons = num_of_params
        output_neurons = num_of_modes*(num_of_nodes+1)
        model = Sequential()
        model.add(Input(shape=(input_neurons,)))
        for i in range(num_of_layers):
            width_of_layer = neurons_per_layer[i]
            model.add(Dense(width_of_layer, activation='relu'))
            if dropout>0:
                model.add(Dropout(dropout))
        model.add(Dense(output_neurons, activation='linear'))
        model.add(Reshape((num_of_modes, num_of_nodes+1)))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,verbose=0)
        return model
    
class CrossValidation():

    def __init__(self, model, freq_errors, mac_errors):
        self.model = model
        self.freq_errors = freq_errors
        self.mac_errors = mac_errors

    def plot(self, display_plot=True):
        model, freq_errors, mac_errors  = self.model, self.freq_errors, self.mac_errors
        fig,ax = plt.subplots(model.num_of_modes,2,sharex='col',figsize=(8,6))
        handles = [Patch(facecolor='#ccebc5', edgecolor='#a8ddb5', label='Error distribution'),
                   plt.Line2D([0], [0], color='#0868ac', label='Meadian'),
                   plt.Line2D([0], [0], color='#4eb3d3', label='95th percentile')]
        for i in range(6):
            ax[i,0].hist(np.abs(freq_errors[:,i]),bins=np.logspace(np.log10(np.min(np.abs(freq_errors))),np.log10(np.max(np.abs(freq_errors))),100),color='#ccebc5',edgecolor='#a8ddb5')
            ax[i,1].hist(np.abs(mac_errors[:,i]),bins=np.logspace(np.log10(np.min(mac_errors)),np.log10(np.max(mac_errors)),100),color='#ccebc5',edgecolor='#a8ddb5')
            ax[i,0].axvline(np.percentile(np.abs(freq_errors[:,i]),50),0,1,c='#0868ac')
            ax[i,0].axvline(np.percentile(np.abs(freq_errors[:,i]),95),0,1,c='#4eb3d3')
            ax[i,1].axvline(np.percentile(mac_errors[:,i],50),0,1,c='#0868ac')
            ax[i,1].axvline(np.percentile(mac_errors[:,i],95),0,1,c='#4eb3d3')
            ax[i,0].set_yticks([])
            ax[i,1].set_yticks([])
            ax[i,0].set_ylabel(f'Mode {i+1}')
        ax[-1,0].set_xscale('log')
        ax[-1,1].set_xscale('log')
        ax[-1,0].set_xlim(np.min(np.percentile(np.abs(freq_errors),5,axis=0)),None)
        ax[-1,1].set_xlim(np.min(np.percentile(mac_errors,5,axis=0)),1)
        ax[-1,0].set_xlabel('Frequency error')
        ax[-1,1].set_xlabel('MAC error')
        fig.suptitle('Cross-validation results')
        fig.set_facecolor('#e6e6e6')
        fig.subplots_adjust(top=0.87,wspace=0.05)
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.87), ncol=3)
        if display_plot:
            plt.show()
        return fig, ax
    
class Measurement():

    def __init__(self, frequencies, eigenvectors, name):
        '''frequencies: np.ndarray with ndim = 1
        eigenvectors: np.ndarray with ndim = 2
        name: string for showing on plots'''
        assert frequencies.ndim==1, 'The dimension of frequencies is incorrect, it should have one dimension - (#modes).'
        assert eigenvectors.ndim==2, 'The dimension of eigenvectors is incorrect, it should have two dimension - (#modes, #nodes).'
        self.num_of_modes = frequencies.shape[0]
        self.num_of_nodes = eigenvectors.shape[1]
        assert eigenvectors.shape[0]==self.num_of_modes, 'Incompatible dimensions of frequencies and eigenvectors.'
        self.frequencies = frequencies
        self.eigenvectors = normalize_eigenvectors(np.array([eigenvectors]))[0]
        self.name = name

    def AutoMAC(self):
        eigenvectors = self.eigenvectors
        automac = MAC_matrix(eigenvectors,eigenvectors)
        return automac
    
    def plot_AutoMAC(self, display_plot=True):
        frequencies = self.frequencies
        name = self.name
        automac = self.AutoMAC()
        fig,ax = array_plot(automac, name, name, frequencies, frequencies, display_plot=display_plot)
        return fig,ax

class Scenario():

    def __init__(self, model, measurement, dofs_model=None, dofs_measurement=None, modes_to_include=None):
        self.model = model
        self.measurement = measurement
        dofs_model = np.arange(model.num_of_nodes) if dofs_model is None else dofs_model
        dofs_measurement = np.arange(measurement.num_of_nodes) if dofs_measurement is None else dofs_measurement
        self.dofs_model = dofs_model
        self.dofs_measurement = dofs_measurement
        assert len(dofs_model)==len(dofs_measurement), 'Number of nodes are different in the model and the measurements. They are expected to be the same.'
        self.num_of_nodes = len(dofs_model)
        self.modes_to_include = np.arange(measurement.num_of_modes) if modes_to_include is None else modes_to_include
        self.num_of_modes = len(self.modes_to_include)
        self.weights = np.ones(self.num_of_modes)

    def add_weights(self, weights):
        assert len(weights)==self.num_of_modes, 'Weights are expected to have the same length as there are modes in the measurement.'
        self.weights = np.array(weights)
        
    def predict_modal_properties(self, parameters):
        dofs_model = self.dofs_model
        dofs_measurement = self.dofs_measurement
        model = self.model
        measurement = self.measurement
        frequencies_measurement = measurement.frequencies
        eigenvectors_measurement = measurement.eigenvectors.copy()[:,dofs_measurement][self.modes_to_include]
        frequencies_model,eigenvectors_model = model.predict(parameters)
        frequencies_model = frequencies_model[0]
        eigenvectors_model = eigenvectors_model[0][:,dofs_model]
        return frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement

    def plot_MAC(self, parameters, show_frequencies=False, display_plot=True, **kwargs):
        frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement = self.predict_modal_properties(parameters)
        mac_matrix = MAC_matrix(eigenvectors_model,eigenvectors_measurement)
        freq_units = kwargs['freq_units'] if 'freq_units' in kwargs.keys() else 'Hz'
        if show_frequencies:
            fig,ax = array_plot(mac_matrix, f'Model [{freq_units}]', f'Measurement [{freq_units}]', ticklabels_1=frequencies_model, ticklabels_2=frequencies_measurement, title='MAC matrix', cbar_label='MAC',display_plot=display_plot)
        else:
            fig,ax = array_plot(mac_matrix, f'Model', f'Measurement', title='MAC matrix', cbar_label='MAC',display_plot=display_plot)
        return fig, ax

    def plot_FMAC(self, parameters, show_offdiagonal=False, display_plot=True, ylim=30, **kwargs):
        frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement = self.predict_modal_properties(parameters)
        mac_matrix = MAC_matrix(eigenvectors_model,eigenvectors_measurement)
        X_modes, Y_frequencies, R_macs, colors = prepare_FMAC_input(mac_matrix, frequencies_model, frequencies_measurement, show_offdiagonal)
        fig,ax = plt.subplots(1,figsize=(6,5))
        ax.grid(axis='y', zorder=-2)
        ax.axhline(0,0,1,c='k',linewidth=1, dashes=(100,0.005), zorder=1)
        ax.set_xlabel('Measured mode')
        ax.set_ylabel('Relative error of FEM frequency [%]')
        ax.set_xticks(np.arange(1,len(Y_frequencies)+1))
        ax.set_axisbelow(True)
        facecolor = kwargs['facecolor'] if 'facecolor' in kwargs.keys() else '#E6E6E6'
        ax.set_facecolor(facecolor)
        sc = plt.scatter(X_modes, Y_frequencies, marker='o',s=R_macs, edgecolors='k',linewidths=1,alpha=1,facecolor=colors,zorder=2)
        plt.ylim((-ylim,ylim))
        cax, _ = cbar.make_axes(ax,shrink=0.4,aspect=30) 
        cb = cbar.ColorbarBase(cax, cmap='GnBu')
        cb.set_ticks([0,  1])
        cb.ax.set_ylabel('MAC value', labelpad=-5)
        if display_plot:
            plt.show()
        return fig, ax, sc, cax
    
class OptimisationSearch():

    def __init__(self, scenario):
        self.scenario = scenario

    def run(self, pop_size, n_offsprings, n_gen):
        problem = OptimisationSearch.OptimizationProblem(self.scenario)
        algorithm = NSGA2(pop_size=pop_size,
                        n_offsprings=n_offsprings,
                        crossover=TwoPointCrossover(),
                        mutation=PolynomialMutation(),
                        eliminate_duplicates=True,
                        save_history=True)
        result = minimize(problem, algorithm,
                    ("n_gen", n_gen),
                    verbose=False)
        self.result=result
        history_parameter = np.zeros((n_gen, pop_size,self.scenario.model.num_of_params))
        history_objective = np.zeros((n_gen, pop_size, problem.n_obj))
        for i in range(history_parameter.shape[0]):
            for j in range(history_parameter.shape[1]):
                history_parameter[i,j] = result.history[i].pop[j].X
        for i in range(history_objective.shape[0]):
            for j in range(history_objective.shape[1]):
                history_objective[i,j] = result.history[i].pop[j].F
        self.history_parameter = history_parameter
        self.history_objective = history_objective
        self.pareto_parameter = result.X
        self.pareto_objective = result.F
        self.evaluations_parameter = np.array(problem.evaluations_parameter)
        self.evaluations_freq_error = np.array(problem.evaluations_freq_error)
        self.evaluations_mac_error = np.array(problem.evaluations_mac_error)
        self.evaluations_objective = np.array(problem.evaluations_objective)

    def plot_pareto_front(self,display_plot=True,xlim=None,ylim=None):
        fig,ax = plt.subplots(1,1)
        plt.scatter(np.array(self.history_objective)[:,:,0].flatten(),np.array(self.history_objective)[:,:,1].flatten(),s=5,facecolor='#ccebc5',edgecolors='#a8ddb5', label='All evaluations')
        plt.plot(*np.transpose(self.pareto_objective[np.argsort(self.pareto_objective[:,0])]),c='#0868ac',linewidth=3,label='Pareto front')
        plt.xlabel('Objective 1: mean relative frequency error')
        plt.ylabel('Objective 2: mean MAC error')
        plt.title('Optimisation results')
        fig.set_facecolor('#e6e6e6')
        plt.legend()
        plt.xlim((0,xlim))
        plt.ylim((0,ylim))
        if display_plot:
            plt.show()
        return fig,ax

    def plot_convergence(self, display_plot=True):
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(7,4))
        n_gen = self.history_parameter.shape[0]
        ax[0].plot(np.arange(n_gen),np.min(self.history_objective,axis=1)[:,0],c='#084081',label='mean relative frequency error')
        ax[0].set_ylabel('Objective 1')
        ax[1].plot(np.arange(n_gen),np.min(self.history_objective,axis=1)[:,1],c='#2b8cbe',label='mean MAC error')
        ax[1].set_ylabel('Objective 2')
        ax[-1].set_xlabel('Generation')
        fig.suptitle('Convergence of objective function')
        fig.set_facecolor('#e6e6e6')
        ax[1].legend()
        if display_plot:
            plt.show()
        return fig,ax

    def plot_parameter_evolution(self, display_plot=True):
        model = self.scenario.model
        nrows = len(model.parameter_names)
        fig, ax = plt.subplots(nrows=nrows,ncols=1,sharex=True,figsize=(7,1.5*nrows+0.5))
        n_gen = self.history_parameter.shape[0]
        if nrows == 1:
            ax = np.array([ax],dtype=object)
        for i in range(nrows):
            ly = model.parameter_bounds[i][0]
            uy = model.parameter_bounds[i][1]
            ax[i].fill_between(np.arange(1,n_gen+1),np.min(self.history_parameter,axis=1)[:,i],np.max(self.history_parameter,axis=1)[:,i],color='#ccebc5',edgecolor='#a8ddb5',label='Full range')
            ax[i].set_ylabel(model.parameter_names[i])
            ax[i].set_ylim((-1,1))
            ax[i].plot(np.arange(1,n_gen+1),self.history_parameter[np.arange(n_gen),np.argmin(self.history_objective[:,:,0],axis=1),i],c='#084081', label='Optimal (frequency)')
            ax[i].plot(np.arange(1,n_gen+1),self.history_parameter[np.arange(n_gen),np.argmin(self.history_objective[:,:,1],axis=1),i],c='#2b8cbe', label='Optimal (MAC)')
            # axes[i].vlines(n_gen+9,np.min(res.X,axis=0)[i],np.max(res.X,axis=0)[i],color='C3', linewidth=5)
            # axes[i].scatter(n_gen-1,res.X[-4,i],color='C3')
            ax_normalised = ax[i].twinx()
            ax_normalised.set_ylim((ly,uy))
            if model.parameter_logarithmic[i]:
                ax_normalised.set_yscale('log')
            # ax_normalised.set_yticks([ly,uy])
            # axes[i].set_yticklabels(axes[i].get_yticks(), rotation=45)
        ax[-1].set_xlim((1,n_gen))
        fig.suptitle('Parameter evelution')
        top_position = 1.5*nrows/(1.5*nrows+0.5)-0.025
        fig.subplots_adjust(top=top_position,wspace=0.05)
        ax[-1].set_xlabel('Generation')
        fig.set_facecolor('#e6e6e6')
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, top_position), ncol=len(handles))
        if display_plot:
            plt.show()
        return fig,ax
    
    def plot_parameter_scatter(self, N=1000, kind='scatter', display_plot=True):
        '''kind (string) - 'scatter' for seaborn.scatterplot or 'kde' for seaborn.kdeplot'''
        model = self.scenario.model
        num_of_params = len(model.parameter_names)
        data = pd.DataFrame(model._to_dimensionless(self.evaluations_parameter[-N:])[:,:num_of_params], columns=model.parameter_names[:num_of_params])
        if kind=='kde':
            plot_kws={'color':'#084081','fill':False,'levels':5}
        elif kind=='scatter':
            plot_kws={'color':'#084081','s':10}
        g = sns.pairplot(data, height=1.3, aspect=1, kind=kind, diag_kind='kde', diag_kws={'color':'#084081'},plot_kws=plot_kws)
        for i in range(num_of_params):
            for j in range(num_of_params):
                g.axes[i,j].set_xlim((-1,1))
                g.axes[i,j].set_ylim((-1,1))
        for ax in g.axes.flatten():
            ax.set_axisbelow(True)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
        g.fig.set_facecolor('#e6e6e6')
        plt.subplots_adjust(top=1.3*num_of_params/(1.3*num_of_params+0.5)-0.025)
        g.fig.suptitle('Parameter scatterplot')
        for i in range(num_of_params):
            for j in range(num_of_params):
                if i == 0:  # Top row
                    twin_ax_x = g.axes[i, j].twiny()
                    twin_ax_x.set_xlim(model.parameter_bounds[j])
                    if model.parameter_logarithmic[j]:
                        twin_ax_x.set_xscale('log')
                if j == num_of_params - 1:  # Last column
                    twin_ax_y = g.axes[i,j].twinx()
                    twin_ax_y.set_ylim(model.parameter_bounds[i])
                    if model.parameter_logarithmic[i]:
                        twin_ax_y.set_yscale('log')
        if display_plot:
            plt.show()
        return g

    class OptimizationProblem(ElementwiseProblem):

        def __init__(self, scenario):
            '''There are two objective functions: one for frequency errors and one for MAC errors. 
            frequency_error = np.sum(np.abs(errors_frequency)*weights)/np.sum(weights)
            MAC_error = np.sum(errors_mac*weights)/np.sum(weights)''' 
            self.scenario = scenario
            parameter_names = scenario.model.parameter_names
            n_var = len(parameter_names)
            self.weights = scenario.weights
            self.n_var = n_var
            self.evaluations_parameter = []
            self.evaluations_freq_error = []
            self.evaluations_mac_error = []
            self.evaluations_objective = []
            n_obj = 2
            self.n_obj = n_obj
            xl = -np.ones(n_var)
            xu = np.ones(n_var)
            super().__init__(n_var=n_var,n_obj=n_obj,n_ieq_constr=0,xl=xl,xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = self.calculate_objective_function(x)

        def calculate_objective_function(self,values):
            scenario = self.scenario
            model = scenario.model
            parameters = model._from_dimensionless(values)
            self.evaluations_parameter.append(parameters[0])
            weights = scenario.weights
            frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement = scenario.predict_modal_properties(parameters)
            frequency_errors, mac_errors = compute_errors(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement)
            objective_frequency, objective_mac = compute_objective_function(frequencies_model, frequencies_measurement, eigenvectors_model, eigenvectors_measurement, weights)
            objective_function = (objective_frequency, objective_mac)
            self.evaluations_freq_error.append(frequency_errors)
            self.evaluations_mac_error.append(mac_errors)
            self.evaluations_objective.append(objective_function)
            return objective_function 
