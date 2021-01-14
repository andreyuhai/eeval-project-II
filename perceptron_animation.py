from manim import *
import numpy as np

# config['quality']='low_quality'
config['save_as_gif'] = True

# NumberPlane configuration
plane_config = dict(
            axis_config = { 
                "include_tip": True, "include_numbers" : True,
                "include_ticks" : True, "line_to_number_buff" : 0.01,
                "stroke_color" : WHITE, "stroke_width": 1,
                "number_scale_val" : 0.7,
                "tip_scale": 0.5
            },
            x_axis_config = {
                "exclude_zero_from_default_numbers": True,
                "label_direction" : DOWN, "stroke_color" : WHITE,
                "x_min" : -10, "x_max" : 9, "unit_size": 1
            },
            y_axis_config = {
                "exclude_zero_from_default_numbers": True,
                "label_direction" : UR, "stroke_color" : WHITE,
                "x_min" : -10, # not y_min
                "x_max" : 10,  # not y_max
                "unit_size": 1
            },
            background_line_style = {
                "stroke_width" : 1, "stroke_opacity" : 1,
                "stroke_color" : GOLD,
            }  
        )

class Foo(Scene):
    def __init__(self,
                 x_pos,
                 y_pos,
                 x_neg,
                 y_neg,
                 weights=np.zeros((3, 1)),
                 bias=0,
                 learning_rate=0.1,
                 **kwargs
                ):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.r = np.max(np.linalg.norm(x, axis=1))
        
        self.x_pos = x_pos
        self.y_pos = y_pos
        
        self.x_neg = x_neg
        self.y_neg = y_neg
        
        self.x = np.vstack((x_pos, x_neg))
        self.y = np.vstack((y_pos, y_neg))
        
        self.weight_vector = None
        self.position_vector = None
        self.decision_boundary = None
        self.weights_text = None
        
        super().__init__()
    
    def construct(self):        
        # Number plane, Positive and Negative samples
        self.np = NumberPlane(**plane_config).scale(0.4).to_edge(buff=0) 
        pos = VGroup(*[Dot(self.np.c2p(*sample)) for sample in self.x_pos])
        neg = VGroup(*[Dot(self.np.c2p(*sample)) for sample in self.x_neg])
        
        pos.set_color(RED)
        neg.set_color(WHITE)
        
        # Pseudo-code
        l1 = MathTex(r"let D=\{(\bar{x}_1, y_1),...,(\bar{x}_l, y_l)\} \subset \mathbb{R}^n \times \{+1, -1\}")
        l2 = MathTex(r"let~0 < \eta < 1").move_to(l1.get_corner(LEFT) + DOWN / 2, LEFT)
        l3 = MathTex(r"\bar{w} \leftarrow \bar{0}").move_to(l2.get_corner(LEFT) + DOWN / 2, LEFT)
        l4 = MathTex(r"b \leftarrow 0").move_to(l3.get_corner(LEFT) + DOWN / 2, LEFT)
        l5 = MathTex(r"r \leftarrow max\{|\bar{x}|~|~(\bar{x}, y) \in D\}").move_to(l4.get_corner(LEFT) + DOWN / 2, LEFT)
        l6 = MathTex(r"repeat").move_to(l5.get_corner(LEFT) + DOWN / 2, LEFT)
        l7 = MathTex(r"for~i = 1~to~l").move_to(l6.get_corner(LEFT) + DOWN / 2, LEFT).shift(RIGHT / 2)
        l8 = MathTex(r"if~sgn(\bar{w} \cdot \bar{x}_{i} - b) \neq y_{i}~then").move_to(l7.get_corner(LEFT) + DOWN / 2, LEFT).shift(RIGHT / 2)
        l9 = MathTex(r"\bar{w} \leftarrow \bar{w} + \eta y_{i} \bar{x}_i").move_to(l8.get_corner(LEFT) + DOWN / 2, LEFT).shift(RIGHT / 2)
        l10 = MathTex(r"b \leftarrow b - \eta y_{i} r^2").move_to(l9.get_corner(LEFT) + DOWN / 2, LEFT)
        l11 = MathTex(r"end~if").move_to(l8.get_corner(LEFT) + 3 * DOWN / 2, LEFT)
        l12 = MathTex(r"end~for").move_to(l7.get_corner(LEFT) + 5 * DOWN / 2, LEFT)
        l13 = MathTex(r"until~sgn(\bar{w} \cdot \bar{x}_{j} - b) = y_j ~ with ~ j = 1,...,l").move_to(l6.get_corner(LEFT) + 7 * DOWN / 2, LEFT)
        l14 = MathTex(r"return~(\bar{w}, b)").move_to(l13.get_corner(LEFT) + DOWN / 2, LEFT)
        
        pseudo_code = VGroup(l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14).scale(0.6).to_corner(UR)

        # Eta and r Trackers
        eta_tracker = Variable(self.learning_rate, MathTex(r"\eta"))
        r_tracker   = Variable(self.r, MathTex(r"r"))

        parameter_t_group = VGroup(eta_tracker, r_tracker).scale(0.6).arrange(RIGHT, buff=LARGE_BUFF).align_to(l14, DL).shift(DOWN)

        # Weight & Bias Trackers
        w1_tracker = Variable(self.weights[0], MathTex(r"w_1"))
        w2_tracker = Variable(self.weights[1], MathTex(r"w_2"))
        bias_tracker = Variable(self.bias, MathTex(r"b"))
        weights_t_group = VGroup(w1_tracker, w2_tracker, bias_tracker).scale(0.6).arrange(RIGHT, buff=LARGE_BUFF).align_to(parameter_t_group, DL).shift(DOWN / 2)
        
        w1_tracker.add_updater(lambda obj: obj.tracker.set_value(self.weights[0]))
        w2_tracker.add_updater(lambda obj: obj.tracker.set_value(self.weights[1]))
        bias_tracker.add_updater(lambda obj: obj.tracker.set_value(self.bias))

        # Sample (x1, x2, y) Trackers
        x1_tracker = Variable(None, MathTex(r"x_1"))
        x2_tracker = Variable(None, MathTex(r"x_2"))
        y_tracker  = Variable(None, MathTex(r"y"))

        sample_t_group = VGroup(x1_tracker, x2_tracker, y_tracker).scale(0.6).arrange(RIGHT, buff=LARGE_BUFF).align_to(weights_t_group, DL).shift(DOWN / 2)

        # Sign Tracker
        sgn_tracker = Variable(None, MathTex(r"sgn(\bar{w} \cdot \bar{x}_i - b)")).scale(0.6).align_to(sample_t_group, DL).shift(DOWN / 2)

        # ==== A N I M A T I O N ====
        
        # Pseudo-code
        self.play(Write(pseudo_code))
        
        # Arrow pointing to pseudo-code in each step
        step_arrow = Arrow(ORIGIN, RIGHT).set_color(YELLOW).next_to(l1, LEFT)

        # Numberplane & points creation
        self.play(Write(self.np))
        self.play(ShowCreation(step_arrow))
        self.play(ShowCreation(pos))
        self.play(ShowCreation(neg))


        # Write eta tracker
        self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l2, RIGHT)))
        self.play(Write(eta_tracker))

        # Write weight trackers
        self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l3, RIGHT)))
        self.play(Write(weights_t_group[:2]))

        # Write bias tracker
        self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l4, RIGHT)))
        self.play(Write(bias_tracker))

        # Write r tracker
        self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l5, RIGHT)))
        self.play(Write(r_tracker))

        # Go over repeat
        self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l6, RIGHT)))

        is_sample_t_group_on_screen = False
        is_sgn_on_screen = False

        while True:
            # Go over for 
            self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l7, RIGHT)))
            for idx, sample in enumerate(x):
                if not is_sample_t_group_on_screen:
                    x1_tracker.add_updater(lambda obj: obj.tracker.set_value(sample[0]))
                    x2_tracker.add_updater(lambda obj: obj.tracker.set_value(sample[1]))
                    y_tracker.add_updater(lambda obj: obj.tracker.set_value(self.y[idx]))

                    self.wait()
                    self.play(Write(sample_t_group))
                    is_sample_t_group_on_screen = True

                self.draw_position_vector(sample)
                self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l8, RIGHT)))
                sgn = np.sign(np.dot(sample, self.weights) - self.bias)
                if not is_sgn_on_screen:
                    sgn_tracker.add_updater(lambda obj: obj.tracker.set_value(sgn))

                    self.wait()
                    self.play(Write(sgn_tracker))
                    is_sgn_on_screen = True

                if sgn != self.y[idx]:
                    self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l9, RIGHT)))
                    self.weights += self.y[idx] * self.learning_rate * sample.reshape(len(sample), 1)
                    
                    self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l10, RIGHT)))
                    self.bias -= self.learning_rate * self.y[idx] * self.r ** 2

                    self.draw_weight_vector()
                    self.draw_decision_boundary()

                self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l11, RIGHT)))

            self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l12, RIGHT)))
            self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).set_color(YELLOW).next_to(l13, RIGHT)))
            if np.all(np.sign(np.dot(self.x, self.weights) - self.bias) == self.y):
                self.play(Transform(step_arrow, Arrow(ORIGIN, LEFT).next_to(l14, RIGHT)))
                break

    def draw_position_vector(self, point):
        if self.position_vector:
            self.play(Transform(self.position_vector, Arrow(self.np.c2p(*ORIGIN), self.np.c2p(*point), buff=0)))
        else:
            self.position_vector = Arrow(self.np.c2p(*ORIGIN), self.np.c2p(*point), buff=0)
            self.play(ShowCreation(self.position_vector))
            
    def draw_weight_vector(self):
        if self.weight_vector:
            self.play(Transform(self.weight_vector, Arrow(self.np.c2p(*ORIGIN), self.np.c2p(*self.weights.T[0]), buff=0)))
        else:
            if np.any(self.weights):
                self.weight_vector = Arrow(self.np.c2p(*ORIGIN), self.np.c2p(*self.weights.T[0]), buff=0)
                self.play(ShowCreation(self.weight_vector))
            
    def draw_decision_boundary(self):
        f = lambda x : float(-(self.weights[0] / self.weights[1]) * x + self.bias / self.weights[1])
        
        if self.decision_boundary:
            self.play(Transform(self.decision_boundary, Line(self.np.c2p(self.np.x_min, f(self.np.x_min), 0), self.np.c2p(self.np.x_max, f(self.np.x_max), 0))))
        else:
            if np.any(self.weights):
                self.decision_boundary = Line(self.np.c2p(self.np.x_min,  f(self.np.x_min), 0), self.np.c2p(self.np.x_max, f(self.np.x_max), 0))
                self.play(ShowCreation(self.decision_boundary))

# Create the training samples
N = 6
x = np.array([[1, 2, 0],
    [3, 3, 0],
    [1.5, -2, 0],
    [-1, -2, 0],
    [-2, -1, 0],
    [-3, -1.5, 0]])

y = np.ones((6,1))
y[3:, :] *= -1

x_pos = x[0:N//2, :]
y_pos = y[0:N//2, :]

x_neg = x[N//2:, :]
y_neg = y[N//2:, :]

foo = Foo(x_pos, y_pos, x_neg, y_neg)
foo.render()
