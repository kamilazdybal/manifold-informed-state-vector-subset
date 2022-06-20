########################################################################
## Load steady laminar flamelet data
########################################################################

species_to_remove_list = ['N2', 'AR', 'HE']

data_path = '../data-sets/'
    
STEADY_state_space = pd.read_csv(data_path + 'STEADY-' + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()
STEADY_state_space_sources = pd.read_csv(data_path + 'STEADY-' + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()
STEADY_mf = pd.read_csv(data_path + 'STEADY-' + data_tag + '-mixture-fraction.csv', sep = ',', header=None).to_numpy()
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
mf = STEADY_mf
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
    mf = mf[idx_sample,:]

    (n_observations, n_variables) = np.shape(state_space)
    
print('\nThe data set has ' + str(n_observations) + ' observations.')

if data_tag =='clustered-flamelet-H2': target_variables_indices = [0, 2, 4, 5, 6]
if data_tag =='clustered-flamelet-CO-H2-10-1': target_variables_indices = [0, 1, 2, 4, 5, 8, 9]
if data_tag =='lightweight-flamelet-CO-H2-10-1': target_variables_indices = [0, 1, 2, 4, 5, 8, 9]
if data_tag =='clustered-flamelet-CH4-gri30': target_variables_indices = [0, 4, 5, 6, 14, 15, 16, 36]
if data_tag =='clustered-flamelet-C2H4': target_variables_indices = [0, 4, 5, 6, 14, 15, 22]
    
target_variables = state_space[:,target_variables_indices]
target_variables_names = list(state_space_names[target_variables_indices])
(_, n_target_variables) = np.shape(target_variables)

print('\nUsing: ' + ', '.join(target_variables_names) + ' as target variables.')