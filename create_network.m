function net = create_network(hidden_layers, hidden_layer_units, input_layer_units, output_layer_units)
% initializes a new network with random parameters

% input_layer_units = 7; % how many input features?
% hidden_layers = 2; % how many hidden layers?
% hidden_layer_units = [8, 8]; % how many nodes per hidden layer?
% output_layer_units = 8; % how many outputs?

% create the network
net = struct();
net.hidden_layers = hidden_layers;
number_of_parameters = [];

% input layer
layer_name = 'W1'; 
bias_name = 'b1';
layer = rand(hidden_layer_units(1), input_layer_units)*2 - 1; % "*2 - 1" sets value range to [-1, +1]
bias = rand(hidden_layer_units(1), 1)*2 - 1;
net.(layer_name) = layer;
net.(bias_name) = bias;

number_of_parameters = [number_of_parameters; size(layer, 1)*size(layer, 2); size(bias, 1)]; % one set for hidden layer, another set for bias

% disp(hidden_layers)

% create hidden layers and output layer
for i = 1:hidden_layers
    % disp(i)
    layer_name = 'W' + string(i+1);
    bias_name = 'b' + string(i+1);

    if i == hidden_layers
        layer = rand(output_layer_units, hidden_layer_units(i))*2 - 1;
        bias = rand(output_layer_units, 1)*2 - 1;
    else
        layer = rand(hidden_layer_units(i+1), hidden_layer_units(i))*2 - 1;
        bias = rand(hidden_layer_units(i+1), 1)*2 - 1;
    end
    number_of_parameters = [number_of_parameters; size(layer, 1)*size(layer, 2); size(bias, 1)];

    net.(layer_name) = layer;
    net.(bias_name) = bias;
end

net.num_parameters = number_of_parameters;

end