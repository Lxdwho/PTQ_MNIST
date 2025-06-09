import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class MYNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=1)  # 28 * 28 * 3
        self.mp1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=3, padding=0, bias=False, groups=1)  # 09 * 09 * 3
        self.cn2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False, groups=1)  # 09 * 09 * 6
        self.mp2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=3, padding=0, bias=False, groups=1)  # 03 * 03 * 6
        self.cn3 = torch.nn.Conv2d(6, 10, kernel_size=3, stride=1, padding=1, bias=False, groups=1)  # 03 * 03 * 10
        self.mp3 = torch.nn.Conv2d(10, 10, kernel_size=3, stride=3, padding=0, bias=False, groups=1)  # 01 * 01 * 10

    def forward(self, x):
        x = torch.nn.functional.relu(self.cn1(x))
        x = torch.nn.functional.relu(self.mp1(x))
        x = torch.nn.functional.relu(self.cn2(x))
        x = torch.nn.functional.relu(self.mp2(x))
        x = torch.nn.functional.relu(self.cn3(x))
        x = self.mp3(x)
        x = torch.nn.functional.log_softmax(x.view(-1, 10), dim=1)
        # x = x.view(-1, 10)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("../", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in tqdm(test_data):
            x = x.to(device)
            y = y.to(device)
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = MYNet().to(device)

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(20):
        for (x, y) in tqdm(train_data):
            x = x.to(device)
            y = y.to(device)
            net.zero_grad()
            output = net.forward(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            n_path = './weights/' + str(epoch + 1) + 'net.pt'
            nh_path = './weights/' + str(epoch + 1) + 'net.pth'
            torch.save(net, n_path)
            torch.save(net.state_dict(), nh_path)
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    torch.save(net, './weights/net.pt')
    torch.save(net.state_dict(), './weights/net.pth')
    for (n, (x, _)) in enumerate(test_data):
        x1 = x.to(device)
        # n = n.to(device)
        if n > 3:
            break
        predict = torch.argmax(net.forward(x1[0]))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
