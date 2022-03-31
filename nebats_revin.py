import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

class ReVIN_normalization(tf.keras.layers.Layer):
    def __init__(self, noise, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise
        self.gamma = tf.Variable(initial_value = 0.1, trainable = True)
        self.beta = tf.Variable(initial_value = 0.1, trainable = True)
        
    
    def call(self,inputs):
        self.mean = tf.math.reduce_mean(inputs)
        self.variance = tf.math.reduce_variance(inputs)
        x = inputs
        x = (x - self.mean)/tf.math.sqrt(self.variance + self.noise)
        #for i in x:
         #   i = (i - self.mean)/tf.math.sqrt(self.variance+self.noise)
        transformed = self.gamma*x + self.beta
        #The below array denorm_array will return the parameters to the denormalisation layer to use.
        denorm_array = [self.mean, self.variance, self.gamma, self.beta, self.noise]
        
        return transformed,  denorm_array


class ReVinDenormalize(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        predictions, denorm_array = inputs
        mean = denorm_array[0]
        variance = denorm_array[1]
        gamma = denorm_array[2]
        beta = denorm_array[3]
        noise = denorm_array[4]
        retransformed = (predictions - beta)/gamma
        denormalized = mean + (retransformed*tf.math.sqrt(variance + noise))

        return denormalized


class NBeatsblock(tf.keras.layers.Layer):
    def __init__(self, input_size, forecast_horizon,no_neurons, no_layers,no_theta_neurons,**kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.no_neurons = no_neurons
        self.no_layers = no_layers
        self.no_theta_neurons = no_theta_neurons
        self.fc_relu = [tf.keras.layers.Dense(self.no_neurons, activation = 'relu')for i in range(self.no_layers)]
        self.theta = tf.keras.layers.Dense(self.no_theta_neurons, activation ="linear")
    def call(self, inputs):
        x = inputs
        for i in self.fc_relu:
            x = i(x)
        theta_output= self.theta(x)
        backcast, forecast = theta_output[:,:self.input_size], theta_output[:, -self.forecast_horizon:]
        return backcast, forecast


class NbeatsStack(tf.keras.layers.Layer):
    def __init__(self,no_of_blocks,input_sp, block_input_size, block_forecast_horizon, no_block_layers, no_theta_neurons, **kwargs):
        self.no_of_blocks = no_of_blocks
        self.input_sp = input_sp
        self.block_input_size = block_input_size
        self.block_forecast_horizon = block_forecast_horizon
        self.no_block_layers = no_block_layers
        self.no_theta_neurons = no_theta_neurons
        self.input_layer = Input(shape = input_sp)
        self.nbeats_block_layer = NBeatsblock(input_size=block_input_size,  forecast_horizon = block_forecast_horizon,
        no_theta_neurons = no_theta_neurons, no_neurons=128,no_layers = no_block_layers )

        super().__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        backcast, forecast = self.nbeats_block_layer(x)
        residuals = tf.keras.layers.subtract([x, backcast])
        for i,j in enumerate(range(self.no_of_blocks)):
            backcast, block_forecast = self.nbeats_block_layer(residuals)
            residuals = tf.keras.layers.subtract([residuals, backcast])
            forecast = tf.keras.layers.add([forecast, block_forecast])

        return residuals, forecast


def get_nbeats_revin_model(input_shape, input_length, forecast_length ):
    theta_neurons_no = input_length+forecast_length
    Input_layer = Input(shape = input_shape)
    #initialising the revin layer
    revin_init = ReVIN_normalization(noise=0.01)
    #Revin layer which is connected to the input layer
    revin_layer, denorm_array = revin_init(Input_layer)
    stack_layer = NbeatsStack(no_of_blocks=4, input_sp  = input_shape, block_input_size=input_length, block_forecast_horizon=forecast_length, no_block_layers=4, no_theta_neurons=theta_neurons_no)
    #creating the first NBeats stack layer which takes normalised inputs from the revin layer
    residuals_stack, forecast = stack_layer(revin_layer)
    #residual connection
    residuals = tf.keras.layers.subtract([revin_layer, residuals_stack])
    #creating 10 more nbeats stacks
    for i in enumerate(range(10)):
        residuals_stack, stack_forecast = NbeatsStack(no_of_blocks=4, input_sp = input_shape, block_input_size=input_length, block_forecast_horizon=forecast_length, no_block_layers=4, no_theta_neurons=theta_neurons_no)(residuals)
        residuals = tf.keras.layers.subtract([residuals, residuals_stack])
        forecast = tf.keras.layers.add([forecast, stack_forecast])
    #The global forecast from the nbeats stack goes into the denormalisation layer    
    denorm_init = ReVinDenormalize()
    denorm_layer = denorm_init([forecast, denorm_array])
    model_nbeats_revin = Model(inputs = Input_layer, outputs = denorm_layer)

    return model_nbeats_revin
