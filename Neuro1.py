import random

def exp(x, terms=10):
    result = 1
    factorial = 1
    power = 1
    for i in range(1, terms):
        factorial *= i
        power *= x
        result += power / factorial
    return result

def tanh(x):
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def dot_product(w, x):
    return sum(w[i] * x[i] for i in range(len(w)))

def forward_pass(x, w1, w2, b1, b2):
    z1 = [dot_product(w1[i], x) + b1 for i in range(len(w1))]
    a1 = [tanh(z) for z in z1]
    
    z2 = [dot_product(w2[i], a1) + b2 for i in range(len(w2))]
    a2 = [tanh(z) for z in z2]
    
    return a2

random.seed(42)
input_size = 2
hidden_size = 2
output_size = 2
w1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
w2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]

b1 = 0.5
b2 = 0.7

test_input = [0.05, 0.10]

output = forward_pass(test_input, w1, w2, b1, b2)
print("Output of the network:", output)
