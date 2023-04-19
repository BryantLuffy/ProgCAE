import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
class ProgCAE:
    def __init__(
            self,
            input_shape,
            kernel_height1=24,
            kernel_height2=12,
            kernel_height3=6,
            stride1=5,
            stride2=5,
            stride3=4,
            filter1=32,
            filter2=64,
            filter3=128,
            hidden_dim=30,
            optimizer=tf.keras.optimizers.Adam,
            epochs=300,
            batch_size=32,
            validation_rate=0.2,
            learning_rate=0.0005
    ):
        """
        A convolutional autoencoder class that initializes its hyperparameters.

        Args:
            input_shape (int): The shape of the input data.
            kernel_height1 (int): The height of the first convolutional layer's kernel. Default is 24.
            kernel_height2 (int): The height of the second convolutional layer's kernel. Default is 12.
            kernel_height3 (int): The height of the third convolutional layer's kernel. Default is 6.
            stride1 (int): The stride of the first convolutional layer. Default is 5.
            stride2 (int): The stride of the second convolutional layer. Default is 5.
            stride3 (int): The stride of the third convolutional layer. Default is 4.
            filter1 (int): The number of filters in the first convolutional layer. Default is 32.
            filter2 (int): The number of filters in the second convolutional layer. Default is 64.
            filter3 (int): The number of filters in the third convolutional layer. Default is 128.
            hidden_dim (int): The dimension of the hidden layer. Default is 30.
            optimizer (tf.keras.optimizers): The optimizer for training the model. Default is Adam.
            epochs (int): The number of epochs to train the model for. Default is 300.
            batch_size (int): The batch size for training the model. Default is 32.
            validation_rate (float): The proportion of data to use for validation. Default is 0.2.
            learning_rate (float): The learning rate for the optimizer. Default is 0.0005.
        """
        # Initialize hyperparameters
        self.kernel_height1=kernel_height1
        self.kernel_height2 = kernel_height2
        self.kernel_height3 = kernel_height3
        self.stride1=stride1
        self.stride2 = stride2
        self.stride3 = stride3
        self.filter1=filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.hidden_dim=hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.learning_rate = learning_rate

        # Create a sequential model
        self.model = Sequential()

        # Add convolutional layers to the model
        self.model.add(Conv2D(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu',
                              input_shape=(1, input_shape, 1)))
        self.model.add(Conv2D(filter2, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        self.model.add(Conv2D(filter3, (1, kernel_height3), strides=(1, stride3), activation='relu', padding='same'))

        # Flatten the output of the convolutional layers
        self.model.add(Flatten())

        # Add a dense layer for the hidden representation
        self.model.add(Dense(units=hidden_dim, name='HiddenLayer'))

        # Add a dense layer for the output
        self.model.add(Dense(units=filter3 * int(input_shape / (stride3 * stride2 * stride1)), activation='relu'))
        self.model.add(Reshape((1, int(input_shape / (stride3 * stride2 * stride1)), filter3)))

        ## Add deconvolutional layers to the model
        self.model.add(
            Conv2DTranspose(filter2, (1, kernel_height3), strides=(1, stride3), padding='same', activation='relu'))
        self.model.add(
            Conv2DTranspose(filter1, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        self.model.add(Conv2DTranspose(1, (1, kernel_height1), strides=(1, stride1), activation='relu', padding='same'))

        self.optimizer = optimizer

    def fit(self, data):
        # Compile the model with mean squared error loss and an optimizer with the learning rate set to the instance's value
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer(lr=self.learning_rate))
        # Reshape the input data to the expected shape
        data_train = tf.reshape(data, [-1, 1, data.shape[1], 1])
        # Train the model on the reshaped input data
        self.model.fit(data_train, data_train,
                       epochs=self.epochs,  # Number of times to iterate over the data
                       batch_size=self.batch_size,  # Number of samples per gradient update
                       validation_split=self.validation_rate)  # Fraction of the data to use for validation during training

    def extract_feature(self, x):
        # Get a Keras backend function that extracts the output of the 'HiddenLayer' layer given an input
        f = tf.keras.backend.function(self.model.input, self.model.get_layer('HiddenLayer').output)
        # Reshape the input data to the expected shape
        x = tf.reshape(x, [-1, 1, x.shape[1], 1])
        # Use the backend function to extract the hidden layer output for the input data
        return f(x)

