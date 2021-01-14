## Single Perceptron Training Animation

![perceptron-training](./perceptron_training.gif)

## Example Usage

```python
# animation.py

from perceptron_animation import PerceptronAnimation

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

p = PerceptronAnimation(x_pos, y_pos, x_neg, y_neg)
p.render()
```

Then simply do `python animation.py` which will render the animation.

## Notes

Algorithm shown in the video/gif is taken from the book Knowledge Discovery with Support Vector Machines by Lutz Hamel.

Animated by using the library `[ManimCommunity/manim](https://github.com/ManimCommunity/manim)`
