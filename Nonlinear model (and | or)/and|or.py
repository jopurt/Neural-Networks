import copy
import random
import math

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0, 0, 0, 1]

# W11 = random.uniform(0,1)
# W12 = random.uniform(0,1)
# B1 = random.uniform(0,1)

W11 = random.normalvariate(0, 0.1)
W12 = random.normalvariate(0, 0.1)
B1 = random.normalvariate(0, 0.1)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print('W11:',W11)
print('W12:',W12)
print('B1:',B1)


def forward_loss(X, Y, W11, W12, B1):
    M = copy.deepcopy(X)

    for i in range(len(X)):
        M[i][0] = M[i][0] * W11
        M[i][1] = M[i][1] * W12
        M[i] = sum(M[i])

    P = copy.deepcopy(M)

    for i in range(len(X)):
        P[i] += B1
        print('before sigmoid:',P[i])

        P[i] = sigmoid(P[i])
        print('after sigmoid:',P[i])

    loss = 0

    for i in range(len(X)):
        loss += pow(Y[i] - P[i], 2)
    loss /= len(X)

    forward_info = {}

    forward_info['X'] = X
    forward_info['Y'] = Y
    forward_info['M'] = M
    forward_info['P'] = P

    return forward_info, loss

forward_info, loss = forward_loss(X, Y, W11, W12, B1)

def loss_gradients(forward_info, W11, W12, B1):

    dLdP = []
    dPdW11 = []
    dPdW12 = []
    dPdB1 = []
    dLdW11 = []
    dLdW12 = []
    dLdB1 = []

    for i in range(len(forward_info['X'])):

        dLdP.append(-2 * (forward_info['Y'][i] - forward_info['P'][i]))

        dPdW11.append(forward_info['X'][i][0])
        dPdW12.append(forward_info['X'][i][1])

        dPdB1.append(1)

        dLdW11.append(dLdP[i] * dPdW11[i])
        dLdW12.append(dLdP[i] * dPdW12[i])

        dLdB1.append(dLdP[i] * dPdB1[i])

    loss_info = {}
    loss_info['W11'] = dLdW11
    loss_info['W12'] = dLdW12
    loss_info['B1'] = dLdB1
    loss_info['P'] = dLdP

    return loss_info

def train(X, Y, epochs, learning_rate, W11, W12, B1):

    for epoch in range(epochs+1):
        print('W11:',W11)
        print('W12:',W12)
        print('B1:',B1)

        forward_info, loss = forward_loss(X, Y, W11, W12, B1)

        # Calculate grads
        grad_info = loss_gradients(forward_info, W11, W12, B1)

        # Sum grads
        grad_W11 = sum(grad_info['W11']) / len(X)
        grad_W12 = sum(grad_info['W12']) / len(X)
        grad_B1 = sum(grad_info['B1']) / len(X)

        # Updating parameters
        W11 -= learning_rate * grad_W11
        W12 -= learning_rate * grad_W12
        B1 -= learning_rate * grad_B1


        # W11 = float(f"{W11:.{5}f}")
        # W12 = float(f"{W12:.{5}f}")
        # B1 = float(f"{B1:.{5}f}")



        # print(f"Epoch {epoch}, Loss: {loss}, W11: {W11}, W12: {W12}, B: {B1}")
        print(f"Epoch {epoch}, Loss: {loss:.5f}, W11: {W11:.5f}, W12: {W12:.5f}, B: {B1:.5f}")
    return W11, W12, B1

W11, W12, B1 = train(X, Y, 1000, 0.1, W11, W12, B1)

def predict_class(x1, x2, W11, W12, B1):
    prediction = W11 * x1 + W12 * x2 + B1
    probability = sigmoid(prediction)
    print(probability)

    return round(probability)


print(predict_class(0,0, W11, W12, B1))
print(predict_class(0,1, W11, W12, B1))
print(predict_class(1,0, W11, W12, B1))
print(predict_class(1,1, W11, W12, B1))



