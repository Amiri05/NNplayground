<h1>Neural Network Playground (JavaScript)</h1>

A browser-based neural network visualizer built from scratch in JavaScript. This project trains a small neural network without external ML libraries and provides an interactive interface to explore how models learn different 2D datasets in real time.

**Overview**

This is a minimal implementation of a neural network with full control over training and visualization. It focuses on understanding how models behave during training by exposing internal mechanics such as decision boundaries, loss, and dataset structure.

<h2>Features</h2>
<h3>Dataset generation</h3>
- Circle, XOR, Gaussian, and spiral datasets <br>
- Adjustable noise levels <br>
- Train/test split with interactive controls 
<h3>Neural network implementation</h3>
- Forward pass<br>
- Tanh hidden layer<br>
- Sigmoid output<br>
- Backpropagation <br>
- Gradient descent <br>
<h3>Training visualization</h3>
- Real-time loss tracking <br>
- Epoch counter <br>
- Continuous updates during training

<h3>Decision boundary rendering</h3>
- Probability-based heatmap visualization <br>
- Clear distinction between confident predictions and boundary regions <br>
- Separate styling for train vs test data <br>
<h3>Controls</h3>
- Learning rate <br>
- Number of hidden neurons <br>
- Dataset type <br>
- Noise <br>
- Train/test split <br>
- Toggle training data visibility <br>
<h2>Key Challenges</h2>
- Ensuring correct backpropagation by avoiding weight mutation during gradient computation<br>
- Fixing rendering order issues that caused the decision boundary to disappear<br>
- Handling dataset generation edge cases (e.g., Gaussian collapse at zero noise)<br>
- Stabilizing spiral dataset behavior under noise<br>
- Moving from hard classification visuals to probability-based rendering<br>

<h2>Why this project</h2>

Most ML tools abstract away the training process. This project focuses on implementing the full pipeline manually to better understand:<br>

- How gradients flow<br>
- How decision boundaries evolve<br>
- How dataset structure affects learning<br>

Everything is implemented from first principles, without relying on ML frameworks.

<h2>Outcome</h2>

A working interactive application that allows users to experiment with neural networks and observe how they learn different datasets in real time.

<h2>Next steps</h2>
- Extend to deeper architectures<br>
- Add more activation functions<br>
- Improve performance for larger datasets<br>
- Add export/import for trained models<br>
