require 'nn';
require 'svm';

torch.setnumthreads(4)
print('<torch> set nb threads to ', torch.getnumthreads())


-- TODO:
-- Implement a function that takes a training and validation set and a range of
-- parameters C and G The output should be indices i, j, such that C[i], G[i]
-- were the best parameters for accuracy on the validation set.
--
function grid_search(train_data, validation_data, C, G) 
end

-- Load the datasets
function load_data()
	--TODO:
	-- You can change these sizes if your program is taking too long to run
	small_train_size = 2000
	small_validation_size = 2000
	medium_train_size = 10000
	medium_validation_size = 2000
	large_train_size = 48000
	large_validation_size = 12000

	full_train_data = svm.ascread('practical4-data/train_data')
	full_test_data = svm.ascread('practical4-data/test_data')

	small_train_data = nn.NarrowTable(1, small_train_size):forward(full_train_data)
	small_validation_data = nn.NarrowTable(small_train_size + 1, small_validation_size):forward(full_train_data)

	medium_train_data = nn.NarrowTable(1, medium_train_size):forward(full_train_data)
	medium_validation_data = nn.NarrowTable(medium_train_size + 1, medium_validation_size):forward(full_train_data)

	large_train_data = nn.NarrowTable(1, large_train_size):forward(full_train_data)
	large_validation_data = nn.NarrowTable(large_train_size + 1, large_validation_size):forward(full_train_data)
end

print('Loading data ..')
load_data()
print('Finished loading data.\n\n')

-- Use these to do grid search on the small training and validation sets
-- Set up C_small and G_small
C_small = {}
G_small = {}
 
C_small[1] = 2^(-5)
G_small[1] = 2^(-13)
for i=2, 10 do
	C_small[i] = C_small[i-1] * 4
 	G_small[i] = G_small[i-1] * 4
end


print('Performing grid search: Large grid; small dataset\n')
--TODO:
-- Add code here

-- Use these to do grid search on the medium training and validation sets
--C_medium = {}
--G_medium = {}
print('\nPerforming grid search: Medium grid; medium dataset\n')
--TODO:
-- Add code here

-- Use these to do grid search on the large training and validation sets
-- C_large = {}
-- G_large = {}
print('\nPerforming grid search: Small grid; large dataset\n')
--TODO:
-- Add code here

-- TODO:
-- best_C = 
-- best_gamma = 
-- TODO: uncomment below to get results of the full data
--print('\n\nTraining on full training data with best parameters picked by grid search.\n\n')
--flags= string.format('-q -c %f -g %f', best_C, best_gamma)
--model = libsvm.train(full_train_data, flags)
--libsvm.predict(full_test_data, model)
