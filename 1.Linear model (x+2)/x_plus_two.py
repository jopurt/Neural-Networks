import random

X = [1,2,3,4,5,6,7,8,9,10,14,15,16,17,19,20,22,23]
Y = list(map(lambda x:x+2, X))

B = random.uniform(0,2)
print("B",B)

A = random.uniform(1,2)
print("A",A)

# X.remove(3)
# print(X)

# train/test 70/30
def split(X,Y):
    train_X = X.copy()
    test_X = X.copy()
    train_Y = Y.copy()
    test_Y = Y.copy()

    dict_train = {}
    for i in range(len(train_X)):
        dict_train[train_X[i]] = train_Y[i]
    # print(dict_train)

    dict_test = {}
    for i in range(len(test_X)):
        dict_test[test_X[i]] = test_Y[i]
    # print(dict_test)

    for i in range(int(len(train_X) * 0.3)):
        rand = random.choice(train_X)
        train_X.remove(rand)

    for i in train_X:
        test_X.remove(i)

    for i in range(len(test_X)):
        dict_train.pop(test_X[i])

    for i in range(len(train_X)):
        dict_test.pop(train_X[i])

    train_Y.clear()
    for i in dict_train.items():
        train_Y.append(i[1])
        # print(i[1])

    test_Y.clear()
    for i in dict_test.items():
        test_Y.append(i[1])
        # print(i[1])

    return train_X, train_Y, test_X, test_Y

# print(split(X,Y))

train_X, train_Y, test_X, test_Y = split(X,Y)

print(train_X, train_Y, test_X, test_Y)

# def activation_func_relu(x):
#     return max(0, x)

# ax+b
def forward_loss(X, Y, B, A):
    M = list(map(lambda x:x*A, X))
    P = list(map(lambda x:x+B, M))

    # loss MSE (Σ(Yi-Pi)^2) / n
    loss = 0

    for i in range(len(X)):
        loss += pow(Y[i] - P[i], 2)
    loss /= len(X)

    forward_info = {}

    forward_info['X'] = X
    forward_info['M'] = M
    forward_info['P'] = P
    forward_info['Y'] = Y

    return forward_info, loss

# print(forward_loss(train_X, train_Y, B, A))

forward_info, loss = forward_loss(train_X, train_Y, B, A)
print(forward_info)
print('loss:',loss)

def loss_gradients(forward_info,B, A):

    dLdP = []
    dPdA = []
    dPdB = []
    dLdB = []
    dLdA = []
    dLdX = []

    for i in range(len(forward_info['X'])):
        # L = loss(MSE) = (Σ(Yi-Pi)^2) / n
        # Li = (Yi-Pi)^2
        # L1 = (Y1-P1)^2)

        # dLdP = ( (Y-P)^2 )' = -2(Y-P)
        dLdP.append(-2 * (forward_info['Y'][i] - forward_info['P'][i]))

        # P = ax+b ; dPdA = x
        dPdA.append(forward_info['X'][i])

        # P = ax+b ; dPdB = 1
        dPdB.append(1)

        # dLdA = dLdP * dPdA = -2(Y-P) * x
        dLdA.append(dLdP[i] * dPdA[i])

        # dLdB = dLdP * dPdB = -2(Y-P) * 1
        dLdB.append(dLdP[i] * dPdB[i])

        dLdX.append(dLdP[i] * A)

    loss_info = {}

    loss_info['A'] = dLdA
    loss_info['B'] = dLdB
    loss_info['P'] = dLdP
    loss_info['X'] = dLdX

    return loss_info

def train(X, Y, epochs, learning_rate, B, A):
    for epoch in range(epochs+1):
        print('B:',B)
        print('A:',A)
        forward_info, loss = forward_loss(X, Y, B, A)

        # Calculate grads
        grad_info = loss_gradients(forward_info, B, A)

        # Sum grads
        grad_A = sum(grad_info['A']) / len(X)
        grad_B = sum(grad_info['B']) / len(X)

        # Updating parameters
        A -= learning_rate * grad_A
        B -= learning_rate * grad_B
        A = float(f"{A:.{5}f}")
        B = float(f"{B:.{5}f}")

        # if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.5f}, A: {A:.5f}, B: {B:.5f}")
    return A, B

A, B = train(train_X, train_Y, 100, 0.001, B, A)

def linear(x,a,b):
    return a*x + b

def predict(x, A, B):
    return linear(x, A, B)

# predict for 12 (it is not in the initial data)
# wait 14
print(predict(12,A,B))




