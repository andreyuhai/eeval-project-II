from manim import *
import matplotlib.pyplot as plt
import numpy as np


# config.frame_width = 30

# class VectorArrow(Scene):
#    def construct(self):
#       print(config.frame_height)
#       dot = Dot(ORIGIN)
#       arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
#       numberplane = NumberPlane(y_max=7, y_min=-7)
#       origin_text = Text('(0, 0)').next_to(dot, DOWN)
#       tip_text = Text('(2, 2)').next_to(arrow.get_end(), RIGHT)

#       self.play(ShowCreation(numberplane))
#       self.add(numberplane, dot, arrow, origin_text, tip_text)
#       self.wait(5)

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

# Create the matrix
N = 20
M = np.zeros((N, 2))

np.random.seed(317054)

M[:,0] = np.random.normal(size=(20,))
M[0:int(N/2), 1] = np.random.uniform(size=(10,)) + 0.5
M[int(N/2):, 1] = -np.random.uniform(size=(10, )) - 0.5
angle = np.random.normal()

x = np.zeros(M.shape)
x[:, 0] = M[:, 0] * np.cos(angle) - M[:, 1] * np.sin(angle)
x[:, 1] = M[:, 0] * np.sin(angle) + M[:, 1] * np.cos(angle)

x += np.random.normal(size=(1,2))
y = np.sign(M[:, 1].reshape(len(M[:, 1]), 1))

# positive_samples, negative_samples

class Foo(Scene):
    def construct(self):
        self.weights = np.zeros((3, 1))
        self.bias = 0
        r = np.max(np.linalg.norm(np.vstack(self.x_pos, self.x_neg), axis=1))
        learning_rate = 0.1

        # numberplane = NumberPlane(y_min=-10, y_max=10)
        numberplane = NumberPlane()
        # positive_dots = VGroup(*[Dot(point=x * RIGHT + y * UP) for x, y in x[:int(N/2), :]])
        # negative_dots = VGroup(*[Dot(point=x * RIGHT + y * UP) for x, y in x[int(N/2):, :]])
        positive_dots = VGroup(*[Dot(point=x) for point in x[:int(N/2), :]])

        positive_dots.set_color(WHITE)
        negative_dots.set_color(RED)

        self.play(ShowCreation(numberplane))
        self.play(ShowCreation(positive_dots))
        self.play(ShowCreation(negative_dots))
        # self.wait(2)

        self.draw_weight_vector(self.weights)
        while True:
            for idx, sample in enumerate(positive_dots):
                self.play(ShowCreation(Arrow(ORIGIN, sample.get_center(), buff=0)))

                # if np.sign(np.dot(sample.get_center(), self.weights) - self.bias) != y[idx]:


            # for idx, sample in enumerate(negative_dots):



        # w = np.array([1, -2])
        # bias = 1
        # eta = 0.1
        # r = np.max(np.linalg.norm(x, axis=1))
        self.draw_decision_boundary(self.weights, self.bias, numberplane.x_min, numberplane.x_max)

        self.play(ShowCreation(Arrow(ORIGIN, positive_dots[0].get_center(), buff=0)))
        self.wait(5)


    def draw_weight_vector(self, weights):
        self.weight_vector = Arrow(ORIGIN, weights[0] * RIGHT + weights[1] * UP, buff=0)
        self.play(ShowCreation(self.weight_vector))

    def update_weight_vector(self, new_weights):
        pass

    def draw_decision_boundary(self, weights, bias, x_min, x_max):
        f = lambda x : -(weights[0] / weights[1]) * x + bias / weights[1]

        self.decision_boundary = Line([ x_min,  f(x_min), 0 ], [ x_max, f(x_max), 0 ])
        self.play(ShowCreation(self.decision_boundary))
        self.wait(1)

    def draw_position_vector(self, point, offset):
        pass
