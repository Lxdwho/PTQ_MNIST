from MNIST_CNN import *


def main():
    net = MYNet().to(device)
    net.load_state_dict(torch.load('./weights/0609-15net.pth'))

    test_data = get_data_loader(is_train=False)
    for (n, (x, _)) in enumerate(test_data):
        x1 = x.to(device)
        if n > 8:
            break
        predict = torch.argmax(net.forward(x1[0]))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction:" + str(int(predict)))
    plt.show()


if __name__ == '__main__':
    main()
