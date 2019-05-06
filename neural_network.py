import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

class NeuralNetwork:  
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create np arrays that have the correct dimensions for weights
        self.hidden_weights = np.random.rand(self.input_size, self.hidden_size)        
        self.output_weights = np.random.rand(self.hidden_size, self.output_size)
                                
    def sigmoid(self, x): 
        # calculate the sigmoid value of input x. 
        # Used by both the predict and train functions.
        return 1 / ( 1 + np.exp(-x) )

    def predict(self, point): 
        # classify the input point using the given 
        # input-to-hidden-layer weight matrix and the 
        # given hidden-to-output-layer weights.
                      
        # Calculate hidden node values
        hidden_values = np.dot(point, self.hidden_weights)
        self.hidden_point = self.sigmoid(hidden_values)
        
        # Calculate output node value/s
        output_value = np.dot(self.hidden_point, self.output_weights) 
        return self.sigmoid(output_value)           

    def train(self, point, target_output, learning_rate): 
        # perform a classification step and then a 
        # backpropagation update to all weights using 
        # the given point and target label.
        
        predicted_output = self.predict(point)
        
        output_error = target_output - predicted_output 
        output_change = (learning_rate * output_error * predicted_output * (1 - predicted_output))  
            
        # Update weights for each output node individually
        for weight in output_change:
            for i in range(len(self.output_weights)):  
                self.output_weights[i] += self.hidden_point.T.dot(weight)[i]
                
        hidden_change = []
        # Calculate weight update for each hidden node individually
        for i in range(len(self.hidden_point)):
            hidden_change.append((1 - self.hidden_point)[i] * output_change.dot(self.output_weights[i]))
        
        hidden_change = np.array(hidden_change)
        
        # Update weights for each hidden node individually
        for i in range(len(self.hidden_weights)):  
            self.hidden_weights[i] += hidden_change.dot(point[i])
        
    def epoch(self, training_set, training_labels, learning_rate): 
        # perform a complete training epoch on the given 
        # set of training points with their associated labels. 
        # Basically a loop that wraps around the train method.
        for point, label in zip(training_set, training_labels):
#             print("Input: " + str(point))
#             print("Expected: " + str(label)) 
#             print("Predicted: " + str(self.predict(point))) 
            self.train(point, label, learning_rate)

    def evaluate(self, testing_set, testing_labels): 
        # predict the class (without performing any training) 
        # on the given set of test points and report the fraction 
        # that are classified correctly. This is the final function 
        # used to evaluate the performance of the training network.
        correct = 0
                
        for point, label in zip(testing_set, testing_labels):
            output = self.predict(point)
            output_index = np.argmax(output)
            
            if output[output_index] > 0.5:
                output = 1
            else:
                output = 0

            if output == label:
                correct += 1
                
        return int(correct / len(testing_set) * 100)
    


def xor():
    points = np.array([[0,0], [1,0], [0,1], [1,1]])
    labels = np.array([0, 1, 1, 0])
    
    nn = NeuralNetwork(len(points[0]), 4, 1)
    
    
    for i in range(10000):
        points, labels = shuffle(points, labels, random_state=0)
        nn.epoch(points, labels, .1)
        
    print(str(nn.evaluate(points, labels)) + "% Predicted correctly")

def iris():
    data = load_iris()
    data_and_labels = shuffle(list(zip(data.data, data.target)))
        
    training_data = [data_and_labels[i][0] for i in range(120)]
    training_labels = [data_and_labels[i][1] for i in range(120)]
    
    testing_data = [data_and_labels[i][0] for i in range(120, 150)]
    testing_labels = [data_and_labels[i][1] for i in range(120, 150)]
    
    nn = NeuralNetwork(len(data_and_labels[0][0]), 5, 3)
    
    for i in range(2500):
        training_data, training_labels = shuffle(training_data, training_labels, random_state=0)
        nn.epoch(training_data, training_labels, .1)
    
    print(str(nn.evaluate(testing_data, testing_labels)) + "% Predicted Correctly")
  

# Uncomment xor() to run train the model to recognize the xor function
# xor()

# Uncomment iristo run train the model to recognize the iris dataset
iris()
 