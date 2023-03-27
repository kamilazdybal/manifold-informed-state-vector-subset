import numpy as np
import pandas as pd
import time

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# User settings
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

n_components_list = [2,3,4,5]
scalings_list = ['auto', 'pareto', 'vast', 'range', '0to1', '-1to1', 'level', 'max']

sample_percentage = 100
data_tag = 'clustered-flamelet-CO-H2-10-1'
species_to_remove_list = ['N2', 'AR', 'HE']
data_path = '../data-sets/'

random_seed = 100
norm = 'cumulative'
penalty_function = 'log-sigma-over-peak'
power = 1
vertical_shift = 1
bandwidth_values = np.logspace(-7, 3, 200)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load training data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

STEADY_state_space = pd.read_csv(data_path + 'STEADY-' + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()
STEADY_state_space_sources = pd.read_csv(data_path + 'STEADY-' + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()
STEADY_state_space_names = pd.read_csv(data_path + 'STEADY-' + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()

for species_to_remove in species_to_remove_list:

    # Remove species:
    (STEADY_species_index, ) = np.where(STEADY_state_space_names==species_to_remove)
    if len(STEADY_species_index) != 0:
        print('Removing ' + STEADY_state_space_names[int(STEADY_species_index)] + '.')
        STEADY_state_space = np.delete(STEADY_state_space, np.s_[STEADY_species_index], axis=1)
        STEADY_state_space_sources = np.delete(STEADY_state_space_sources, np.s_[STEADY_species_index], axis=1)
        STEADY_state_space_names = np.delete(STEADY_state_space_names, np.s_[STEADY_species_index])
    else:
        pass

state_space = STEADY_state_space
state_space_sources = STEADY_state_space_sources
state_space_names = STEADY_state_space_names

(n_observations, n_variables) = np.shape(state_space)

if sample_percentage == 100:
    sample_data = False
else:
    sample_data = True

if sample_data:
    idx = np.zeros((n_observations,)).astype(int)
    sample_random = preprocess.DataSampler(idx, random_seed=random_seed, verbose=False)
    (idx_sample, _) = sample_random.random(sample_percentage)

    state_space = state_space[idx_sample,:]
    state_space_sources = state_space_sources[idx_sample,:]

    (n_observations, n_variables) = np.shape(state_space)

print('\nThe data set has ' + str(n_observations) + ' observations.')

if data_tag =='clustered-flamelet-H2': target_variables_indices = [0, 2, 4, 5, 6]
if data_tag =='clustered-flamelet-CO-H2-10-1': target_variables_indices = [0, 1, 2, 4, 5, 8, 9]
if data_tag =='clustered-flamelet-C2H4': target_variables_indices = [0, 4, 5, 6, 14, 15, 22]

target_variables = state_space[:,target_variables_indices]
target_variables_names = list(state_space_names[target_variables_indices])
(_, n_target_variables) = np.shape(target_variables)

print('\nUsing: ' + ', '.join(target_variables_names) + ' as target variables.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Run ranking of scaling criteria across changing manifold dimensionalities 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for n_components in n_components_list:

    print('\nRunning for ' + str(n_components) + 'D LDM\n')

    for scaling in scalings_list:

        print(scaling)

        tic = time.perf_counter()

        pca = reduction.PCA(state_space, scaling=scaling, n_components=n_components)
        PCs = pca.transform(state_space)

        variance_data = analysis.compute_normalized_variance(PCs,
                                                             target_variables,
                                                             depvar_names=target_variables_names,
                                                             scale_unit_box=True,
                                                             bandwidth_values=bandwidth_values)

        cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                     penalty_function=penalty_function,
                                                                     norm=norm,
                                                                     power=power,
                                                                     vertical_shift=vertical_shift,
                                                                     integrate_to_peak=False)

        print('Cumulative cost:\t' + str(round(cost,2)))

        toc = time.perf_counter()

        print(f'\tTime it took: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

        np.savetxt('../results/REPRODUCE-RESULTS-' + data_tag + '-perc-of-data-' + str(sample_percentage) + '-' + str(n_components) + 'D-LDM-optimizing-on-' + '-'.join(target_variables_names) + '-' + scaling + '-scaling-cost.csv', ([cost]), delimiter=',', fmt='%.16e')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Run ranking of scaling with variable selection across changing manifold dimensionalities
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for n_components in n_components_list:

    print('\nRunning for ' + str(n_components) + 'D LDM\n')

    for scaling in scalings_list:

        print(scaling)

        tic = time.perf_counter()

        _, selected_variables, optimized_cost, _ = analysis.manifold_informed_backward_elimination(state_space,
                                                                                state_space_sources,
                                                                                list(state_space_names),
                                                                                scaling=scaling,
                                                                                bandwidth_values=bandwidth_values,
                                                                                target_variables=target_variables,
                                                                                add_transformed_source=False,
                                                                                target_manifold_dimensionality=n_components,
                                                                                penalty_function=penalty_function,
                                                                                norm=norm,
                                                                                integrate_to_peak=False,
                                                                                verbose=True)

        print('Cost of ' + scaling + ' scaling with best subsetting:\t' + str(round(optimized_cost,2)))

        toc = time.perf_counter()

        print(f'\tTime it took: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

        np.savetxt('../results/REPRODUCE-RESULTS-' + data_tag + '-perc-of-data-' + str(sample_percentage) + '-' + str(n_components) + 'D-LDM-optimizing-on-' + '-'.join(target_variables_names) + '-' + scaling + '-scaling-backward-variable-elimination-selected-variables.csv', (selected_variables), delimiter=',', fmt='%.16e')

        np.savetxt('../results/REPRODUCE-RESULTS-' + data_tag + '-perc-of-data-' + str(sample_percentage) + '-' + str(n_components) + 'D-LDM-optimizing-on-' + '-'.join(target_variables_names) + '-' + scaling + '-scaling-backward-variable-elimination-cost.csv', ([optimized_cost]), delimiter=',', fmt='%.16e')
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 