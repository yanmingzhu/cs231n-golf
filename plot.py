import matplotlib.pyplot as plt

def plot(loss, acc):
    loss = [l.detach().item() for l in loss]
    acc = [a.detach().item() for a in acc]

    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(loss))], loss)

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(acc))], acc)

    plt.show()