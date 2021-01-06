from manim import *
import matplotlib.pyplot as plt
import numpy as np


config.frame_width = 30

# y = np.array([[1],
#          [1],
#          [1],
#          [-1],
#          [-1],
#          [-1]])
# x = np.array([[-3,3],
#          [-5, 1],
#          [1, 7],
#          [-1, 1],
#          [2, -1],
#          [3, -2]])


# positive_samples, negative_samples

class Perceptron(Scene):
    # def __init__(self):
    #     self.weight_vector = None
    #     self.position_vector = None
    #     self.decision_boundary = None
    #     self.weights = np.zeros((3, 1))
    #     self.bias = 0

    def construct(self):
        self.weight_vector = None
        self.position_vector = None
        self.decision_boundary = None
        self.weights_text = None
        self.weights = np.zeros((3, 1))
        self.bias = 0
        # self.weights = np.zeros((3, 1))
        # self.bias = 0
        r = np.max(np.linalg.norm(np.vstack((self.x_pos, self.x_neg)), axis=1))
        learning_rate = 0.1

        self.numberplane = NumberPlane(y_min=-10, y_max=10)
        positive_dots = VGroup(*[Dot(point=p) for p in self.x_pos])
        negative_dots = VGroup(*[Dot(point=p) for p in self.x_neg])

        positive_dots.set_color(WHITE)
        negative_dots.set_color(RED)

        self.play(ShowCreation(self.numberplane))
        self.play(ShowCreation(positive_dots))
        self.play(ShowCreation(negative_dots))
        self.wait(1)

        self.show_weights()
        self.draw_weight_vector()
        self.draw_decision_boundary()

        while True:
            for idx, sample in enumerate(np.vstack((self.x_pos, self.x_neg))):
                self.draw_position_vector(sample)
                if np.sign(np.dot(sample, self.weights) - self.bias) != np.vstack((self.y_pos, self.y_neg))[idx]:
                    self.weights += np.vstack((self.y_pos, self.y_neg))[idx] * learning_rate * sample.reshape(len(sample), 1)
                    self.bias -= learning_rate * np.vstack((self.y_pos, self.y_neg))[idx] * r ** 2

                    self.show_weights()
                    print(f"Weights: {self.weights}\nBias: {self.bias}")
                    self.draw_weight_vector()
                    if idx == 0:
                        print("first decision boundary call")
                    self.draw_decision_boundary()
                    if idx == 0:
                        print("done decision boundary call")

            if np.all(np.sign(np.dot(np.vstack((self.x_pos, self.x_neg)), self.weights) - self.bias) == np.vstack((self.y_pos,
                self.y_neg))):
                break


    def draw_weight_vector(self):
        if self.weight_vector:
            # TODO: add offset to ORIGIN
            self.play(Transform(self.weight_vector, Arrow(ORIGIN, self.weights.T[0], buff=0)))
            self.wait(0.5)
        else:
            if np.any(self.weights):
                self.weight_vector = Arrow(ORIGIN, self.weights.T[0], buff=0)
                self.play(ShowCreation(self.weight_vector))
                self.wait(0.5)

    def draw_decision_boundary(self):
        f = lambda x : float(-(self.weights[0] / self.weights[1]) * x + self.bias / self.weights[1])

        if self.decision_boundary:
            print("if self decision boundary girdi")
            self.play(Transform(self.decision_boundary, Line([self.numberplane.x_min,  f(self.numberplane.x_min), 0],
                [self.numberplane.x_max, f(self.numberplane.x_max), 0])))
            self.wait(0.5)
        else:
            if np.any(self.weights):
                print("np.any girdi")
                self.decision_boundary = Line([self.numberplane.x_min,  f(self.numberplane.x_min), 0],
                        [self.numberplane.x_max, f(self.numberplane.x_max), 0])
                print('created')
                self.play(ShowCreation(self.decision_boundary))
                self.wait(1)
            else:
                print("burada")

    def draw_position_vector(self, point):
        if self.position_vector:
            self.play(FadeOut(self.position_vector))
            self.wait(0.5)
            # TODO: Add offset to Arrow ORIGIN
            self.play(Transform(self.position_vector, Arrow(ORIGIN, point, buff=0)))
            self.wait(0.5)
        else:
            # TODO: Add offset to Arrow ORIGIN
            self.position_vector = Arrow(ORIGIN, point, buff=0)
            self.play(ShowCreation(self.position_vector))
            self.wait(0.5)

    def show_weights(self):
        if self.weights_text:
            self.play(Transform(self.weights_text, Text(f"w: ({float(self.weights[0])}, {float(self.weights[1])})\nb: {self.bias}").to_corner(UL)))
        else:
            self.weights_text = Text(f"w: ({float(self.weights[0])}, {float(self.weights[1])})\nb: {self.bias}").to_corner(UL)
            self.play(ShowCreation(self.weights_text))

# Create the matrix
N = 6
M = np.zeros((N, 3))

np.random.seed(317054)

M[:,0] = np.random.normal(size=(N,))
M[0:N//2, 1] = np.random.uniform(size=(N//2,)) + 0.5
M[N//2:, 1] = -np.random.uniform(size=(N//2, )) - 0.5
angle = np.random.normal()

x = np.zeros(M.shape)
x[:, 0] = M[:, 0] * np.cos(angle) - M[:, 1] * np.sin(angle)
x[:, 1] = M[:, 0] * np.sin(angle) + M[:, 1] * np.cos(angle)

x[:, 0:2] += np.random.normal(size=(1,2))
y = np.sign(M[:, 1].reshape(len(M[:, 1]), 1))

x = np.array([[1, 2, 0],
    [3, 3, 0],
    [1.5, -2, 0],
    [-1, -2, 0],
    [-2, -1, 0],
    [-3, -1.5, 0]])

y = np.ones((6,1))
y[3:, :] *= -1

perceptron = Perceptron()
perceptron.x_pos = x[0:N//2, :]
perceptron.y_pos = y[0:N//2, :]

perceptron.x_neg = x[N//2:, :]
perceptron.y_neg = y[N//2:, :]

perceptron.render()
