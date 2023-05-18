function [population, fitness_score, progress] = ecoli_dataset_run()

% load dataset
data = readmatrix('ecoli_csv_cleaned.csv');

input_layer_units = 7; % how many input features?
output_layer_units = 8; % how many outputs?

X = data(:, 1:end-1); % assuming all columns are attributes
Y = data(:, end); % except for the last column for labels

num_hidden_layers = 10; % Maximum number of hidden layers
min_units = 1; % Minimum number of units in a hidden layer
max_units = 10; % Maximum number of units in a hidden layer

% calculate the number of genes
num_genes = (input_layer_units*max_units+max_units)+((max_units*max_units)+max_units)*(num_hidden_layers-1)+(max_units*output_layer_units+output_layer_units);

rng('default');
% random seeds used for ecoli dataset are
% 30, 31, 84
rng(84);

% set parameters for GA
population_size = 500; % 500 chromosomes for population
generations_max = 1000; % run for maximum 100 generations
selrate = 0.2; % SelectionRate
mutrate = 0.3; % MutationRate
progress = [];
  
convergence_maxcount = 10; % stop the GA if the average fitness score stopped increasing for 5 generations
convergence_count = 0;
convergence_avg = 0;

% initialize population
population = rand(population_size, num_genes) * 2 - 1;
% adding two more layers for number of hidden layers and number of hidden units for each layer
population = [population, randi([min_units max_units],[population_size 11])];
    
% initialize an all-zeros row vector
fitness_score = zeros(population_size, 1);
    
generations_current = 1;
while generations_current < generations_max
    % test all chromosomes that haven't been tested
    for i = 1:population_size
        if fitness_score(i,1) == 0
            % fitness testing a chromosome
            fitness_score(i,1) = fitness_function(population(i, 1:end), X, Y, input_layer_units, output_layer_units, num_genes);
        end
    end
    
    % find out statistics of the population
    fit_avg = mean(fitness_score);
    fit_max = max(fitness_score);
    progress = [progress; fit_avg, fit_max];
    
    % convergence? 
    if fit_avg > convergence_avg
        convergence_avg = fit_avg;
        convergence_count = 0;
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    else
        convergence_count = convergence_count + 1;
    end
    
    generations_current = generations_current + 1;
    % stop the GA if reach 100% accuracy or reach convergence
    if (fit_max >= 1.0) % if accuracy 100%
        generations_max = 0;
        disp("Reached convergence.")
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    elseif (convergence_count > convergence_maxcount)
        if (selrate < 0.9)
            convergence_count = 0;
            selrate = selrate + 0.1;
            if mutrate >= 0.1
                mutrate = mutrate - 0.1;
            end
            disp("selrate: "+selrate+", mutrate: "+mutrate);
        else
            generations_max = 0;
            disp("Reached convergence.")
        end
    end
    
    % do genetic operators
    [population, fitness_score] = genetic_operators(population, num_genes, fitness_score, selrate, mutrate);
        

end

% plot a graph fitness score vs generation
x = 1:size(progress, 1);
y = progress;
plot(x, y, '-');
xlabel('Generation');
ylabel('Fitness Score');
legend('Average Fitness', 'Maximum Fitness');
title('Fitness Score vs Generation');

% calculate and display mean and standard deviation
% of hidden layer and hidden layer unit
[hidden_layer_mean, hidden_layer_std, hidden_layer_unit_mean, hidden_layer_unit_std] = calc_mean(population, num_genes);
disp("mean of hidden layers: "+hidden_layer_mean);
disp("std of hidden layers: "+hidden_layer_std);
disp("mean of hidden layer units: ");
disp(hidden_layer_unit_mean);
disp("std of hidden layer units: ");
disp(hidden_layer_unit_std);

end

% fitness function used to calculate the score of a chromosome
function score = fitness_function(chromosome, X, Y, input_layer_units, output_layer_units, num_genes)
% split the population to extract hidden layer and hidden layer unit
[~, hidden_layer, hidden_layer_unit] = split_population(chromosome, num_genes);

net = create_network( ...
    hidden_layer, ...
    hidden_layer_unit, ...
    input_layer_units, ...
    output_layer_units);

layers = (length(fieldnames(net))-2) / 2;
% set the weights based on the chromosome
for i = 1:layers
    layer_name = 'W' + string(i);
    bias_name = 'b' + string(i);
    
    num_genes = size(net.(layer_name), 1)*size(net.(layer_name), 2);
    new_layer = reshape(chromosome(1:num_genes), [size(net.(layer_name), 1), size(net.(layer_name), 2)]);
    net.(layer_name) = new_layer;
    chromosome = chromosome(num_genes+1:end);

    num_genes = size(net.(bias_name), 1);
    new_bias = chromosome(1:num_genes);
    net.(bias_name) = new_bias';
    chromosome = chromosome(num_genes+1:end);
end

% test the new network
Y_pred = test(net, X);

% fitness score is the accuracy of the prediction
score = mean(Y == Y_pred');
% disp(score);

end

function [population, fitness_score] = genetic_operators(population, num_genes, fitness_score, selrate, mutrate)

% calculate the chromosome being rejected based on selection rate
popsize = size(population, 1);
num_reject = round((1-selrate) * popsize);

for i = 1:num_reject
    % find lowest fitness score and remove the chromosome
    [~, lowest] = min(fitness_score);
    population(lowest, :) = [];
    fitness_score(lowest) = [];
end

% for each rejection, create a new chromosome
num_parents = size(population, 1);
for i = 1:num_reject
    % random permutation method
    order = randperm(num_parents);
    parent1 = population(order(1), :);
    parent2 = population(order(2), :);

    % crossing-over to inherit genes randomly from both parents
    offspring = [];
    for j = 1:length(parent1)
        randomValue = rand;
        if randomValue < 0.5
            offspring(j) = parent1(j);
        else
            offspring(j) = parent2(j);
        end
    end

    % mutation
    mut_val = rand(1, size(population(1,:), 2));
    mut_val = mut_val * mutrate;

    for j = 1:size(mut_val, 2)
        if rand < mutrate
            if j < num_genes % if j is within weight and bias portion
                offspring(1, j) = offspring(1, j) + mut_val(1, j);
            else % if j is within hidden layers and hidden layer units portion
                offspring(1, j) = randi([1, 10]);
            end
        end
    end
    
    % add new offspring to population
    population = [population; offspring];
    fitness_score = [fitness_score; 0];
end

end

% function that splits a population into
% weight bias portion, hidden layer portion and hidden unit portion
function [weight_bias_matrix, hidden_layer_matrix, hidden_unit_matrix] = split_population(population, num_genes)

weight_bias_matrix = population(:, 1:num_genes);
hidden_layer_matrix = population(:, num_genes+1);
hidden_unit_matrix = population(:, num_genes+2:end);

end

% function used to calculate mean and standard deviation
function [hidden_layer_mean, hidden_layer_std, hidden_layer_unit_mean, hidden_layer_unit_std] = calc_mean(population, num_genes)

[~, hidden_layer, hidden_layer_unit] = split_population(population, num_genes);

% calculate mean of hidden layers
hidden_layer_mean = mean(hidden_layer);
% calculate std of hidden layers
hidden_layer_std = std(hidden_layer);

pop_size = size(population, 1);

for i = 1:pop_size
    num_hiden_layer = hidden_layer(i);
    row = hidden_layer_unit(i, :);
    row(num_hiden_layer+1:end) = NaN;
    hidden_layer_unit(i, :) = row;
end

% calculate mean of hidden layer units exclude NaN
hidden_layer_unit_mean = nanmean(hidden_layer_unit, 1);
% calculate std of hidden layer units exclude NaN
hidden_layer_unit_std = nanstd(hidden_layer_unit, 1);

end