import copy
import random
import math
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

# W11 = random.uniform(0,1)
# W12 = random.uniform(0,1)
# B1 = random.uniform(0,1)
#
# W21 = random.uniform(0,1)
# W22 = random.uniform(0,1)
# B2 = random.uniform(0,1)
#
# W31 = random.uniform(0,1)
# W32 = random.uniform(0,1)
# B3 = random.uniform(0,1)

# W11 = random.normalvariate(0, 0.1)
# W12 = random.normalvariate(0, 0.1)
# B1 = random.normalvariate(0, 0.1)
#
# W21 = random.normalvariate(0, 0.1)
# W22 = random.normalvariate(0, 0.1)
# B2 = random.normalvariate(0, 0.1)
#
# W31 = random.normalvariate(0, 0.1)
# W32 = random.normalvariate(0, 0.1)
# B3 = random.normalvariate(0, 0.1)

W11 = random.uniform(-1,1)
W12 = random.uniform(-1,1)
B1 = random.uniform(-1,1)

W21 = random.uniform(-1,1)
W22 = random.uniform(-1,1)
B2 = random.uniform(-1,1)

W31 = random.uniform(-1,1)
W32 = random.uniform(-1,1)
B3 = random.uniform(-1,1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


print('W11:', W11)
print('W12:', W12)
print('B1:', B1)

print('W21:', W21)
print('W22:', W22)
print('B2:', B2)

print('W31:', W31)
print('W32:', W32)
print('B3:', B3)


def forward_loss(X, Y, W11, W12, B1, W21, W22, B2, W31, W32, B3):
    M1 = copy.deepcopy(X)
    M2 = copy.deepcopy(X)
    M3 = copy.deepcopy(X)

    # Multiply on weights
    for i in range(len(X)):
        M1[i][0] = M1[i][0] * W11
        M1[i][1] = M1[i][1] * W12
        # M1[i] = sum(M1[i])

        M2[i][0] = M2[i][0] * W21
        M2[i][1] = M2[i][1] * W22
        # M2[i] = sum(M2[i])


    N1 = copy.deepcopy(M1)
    N2 = copy.deepcopy(M2)

    sigN1 = copy.deepcopy(N1)
    sigN2 = copy.deepcopy(N2)

    for i in range(len(X)):
        # # Add bias
        N1[i] = N1[i][0] + N1[i][1] + B1
        sigN1[i] = sigmoid(N1[i])

        N2[i] = N2[i][0] + N2[i][1] + B2
        sigN2[i] = sigmoid(N2[i])


    for i in range(len(X)):
        M3[i][0] = sigN1[i] * W31
        M3[i][1] = sigN2[i] * W32

    N3 = copy.deepcopy(M3)

    for i in range(len(X)):
        N3[i] = N3[i][0] + N3[i][1] + B3

    P = N3.copy()

    for i in range(len(X)):
        P[i] = sigmoid(P[i])

    loss = 0

    for i in range(len(X)):
        loss += pow(Y[i] - P[i], 2)
    loss /= len(X)

    forward_info = {}

    forward_info['X'] = X
    forward_info['Y'] = Y

    forward_info['M1'] = M1
    forward_info['M2'] = M2
    forward_info['M3'] = M3

    forward_info['N1'] = N1
    forward_info['N2'] = N2
    forward_info['N3'] = N3

    forward_info['sigN1'] = sigN1
    forward_info['sigN2'] = sigN2

    forward_info['P'] = P

    return forward_info, loss


# forward_info, loss = forward_loss(X, Y, W11, W12, B1, W21, W22, B2, W31, W32, B3)


def loss_gradients(forward_info, W31, W32):
    dLdP = []
    dPdN3 = []
    dN3dW31 = []
    dN3dW32 = []
    dN3dB3 = []

    dLdW31 = []
    dLdW32 = []
    dLdB3 = []

    dLdsigN1 = []
    dLdsigN2 = []

    dsigN1dN1 = []
    dsigN2dN2 = []

    dN1dW11 = []
    dN1dW12 = []
    dN1dB1 = []

    dN2dW21 = []
    dN2dW22 = []
    dN2dB2 = []

    dLdW11 = []
    dLdW12 = []
    dLdB1 = []

    dLdW21 = []
    dLdW22 = []
    dLdB2 = []


    for i in range(len(forward_info['X'])):
        dLdP.append(-2 * (forward_info['Y'][i] - forward_info['P'][i]))
        dPdN3.append(sigmoid_derivative(forward_info['N3'][i]))


        dN3dW31.append(sigmoid(forward_info['N1'][i]))
        dN3dW32.append(sigmoid(forward_info['N2'][i]))
        dN3dB3.append(1)

        dLdW31.append(dLdP[i] * dPdN3[i] * dN3dW31[i])
        dLdW32.append(dLdP[i] * dPdN3[i] * dN3dW32[i])
        dLdB3.append(dLdP[i] * dPdN3[i] * dN3dB3[i])

        dLdsigN1.append(dLdP[i] * dPdN3[i] * W31)
        dLdsigN2.append(dLdP[i] * dPdN3[i] * W32)

        dsigN1dN1.append(sigmoid_derivative(forward_info['N1'][i]))
        dsigN2dN2.append(sigmoid_derivative(forward_info['N2'][i]))

        dN1dW11.append(forward_info['X'][i][0])
        dN1dW12.append(forward_info['X'][i][1])
        dN1dB1.append(1)

        dN2dW21.append(forward_info['X'][i][0])
        dN2dW22.append(forward_info['X'][i][1])
        dN2dB2.append(1)

        dLdW11.append(dLdsigN1[i] * dsigN1dN1[i] * dN1dW11[i])
        dLdW12.append(dLdsigN1[i] * dsigN1dN1[i] * dN1dW12[i])
        dLdB1.append(dLdsigN1[i] * dsigN1dN1[i] * dN1dB1[i])

        dLdW21.append(dLdsigN2[i] * dsigN2dN2[i] * dN2dW21[i])
        dLdW22.append(dLdsigN2[i] * dsigN2dN2[i] * dN2dW22[i])
        dLdB2.append(dLdsigN2[i] * dsigN2dN2[i] * dN2dB2[i])

    loss_info = {}

    loss_info['W11'] = dLdW11
    loss_info['W12'] = dLdW12
    loss_info['B1'] = dLdB1

    loss_info['W21'] = dLdW21
    loss_info['W22'] = dLdW22
    loss_info['B2'] = dLdB2

    loss_info['W31'] = dLdW31
    loss_info['W32'] = dLdW32
    loss_info['B3'] = dLdB3

    loss_info['P'] = dLdP

    return loss_info


def train(X, Y, epochs, learning_rate, W11, W12, B1, W21, W22, B2, W31, W32, B3):
    for epoch in range(epochs + 1):

        forward_info, loss = forward_loss(X, Y, W11, W12, B1, W21, W22, B2, W31, W32, B3)

        # Calculate grads
        grad_info = loss_gradients(forward_info, W31, W32)

        # Sum grads
        grad_W11 = sum(grad_info['W11']) / len(X)
        grad_W12 = sum(grad_info['W12']) / len(X)
        grad_B1 = sum(grad_info['B1']) / len(X)

        grad_W21 = sum(grad_info['W21']) / len(X)
        grad_W22 = sum(grad_info['W22']) / len(X)
        grad_B2 = sum(grad_info['B2']) / len(X)

        grad_W31 = sum(grad_info['W31']) / len(X)
        grad_W32 = sum(grad_info['W32']) / len(X)
        grad_B3 = sum(grad_info['B3']) / len(X)

        # Updating parameters
        W11 -= learning_rate * grad_W11
        W12 -= learning_rate * grad_W12
        B1 -= learning_rate * grad_B1

        W21 -= learning_rate * grad_W21
        W22 -= learning_rate * grad_W22
        B2 -= learning_rate * grad_B2

        W31 -= learning_rate * grad_W31
        W32 -= learning_rate * grad_W32
        B3 -= learning_rate * grad_B3

        if epoch % 20 ==0:
            print(f"Epoch {epoch}, Loss: {loss},"
                  f" W11: {W11}, W12: {W12}, B1: {B1},"
                  f" W21: {W21}, W22: {W22}, B2: {B2},"
                  f" W31: {W31}, W32: {W32}, B: {B3}")
        # print(f"Epoch {epoch}, Loss: {loss:.5f}, W11: {W11:.5f}, W12: {W12:.5f}, B: {B1:.5f}")
    return W11, W12, B1, W21, W22, B2, W31, W32, B3

W11, W12, B1, W21, W22, B2, W31, W32, B3 = train(X, Y, 100000, 0.1, W11, W12, B1, W21, W22, B2, W31, W32, B3)

def predict_class(x1, x2, W11, W12, B1, W21, W22, B2, W31, W32, B3):
    N1 = W11 * x1 + W12 * x2 + B1
    sigN1 = sigmoid(N1)

    N2 = W21 * x1 + W22 * x2 + B2
    sigN2 = sigmoid(N2)
    
    N3 = W31 * sigN1 + W32 * sigN2 + B3

    prediction = sigmoid(N3)
    print(prediction)

    return round(prediction)

print(predict_class(0, 0, W11, W12, B1, W21, W22, B2, W31, W32, B3))
print(predict_class(0, 1, W11, W12, B1, W21, W22, B2, W31, W32, B3))
print(predict_class(1, 0, W11, W12, B1, W21, W22, B2, W31, W32, B3))
print(predict_class(1, 1, W11, W12, B1, W21, W22, B2, W31, W32, B3))
