import os
import torch
import torch.nn as nn
from demo02 import *
from tqdm import tqdm

device = "cpu"

# 模型打印函数
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p") / 1e3)
    os.remove('temp_delme.p')

# 模型测试函数
def test(model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0
    iterations = 0

    model.eval()

    with torch.no_grad():
        for data in tqdm(get_data_loader(is_train=False), desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break
    print(f'Accuracy: {round(correct / total, 3)}')

# 量化模型
class QuantizedVeryNet(nn.Module):
    def __init__(self):
        super(QuantizedVeryNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        # x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        x = self.fc4(x)
        x = self.dequant(x)
        return x


def main():
    net = Net()
    net.load_state_dict(torch.load('net.pth'))
    # Print the weights matrix of the model before quantization
    print('Weights before quantization')
    print(net.fc1.weight)
    print(net.fc1.weight.dtype)

    print('Size of the model before quantization')
    print_size_of_model(net)

    # 创建量化网络
    net_quantized = QuantizedVeryNet().to('cpu')

    # 从未量化网络中复制权重
    net_quantized.load_state_dict(net.state_dict())
    net_quantized.eval()
    # 装载observer
    net_quantized.qconfig = torch.ao.quantization.default_qconfig
    net_quantized = torch.ao.quantization.prepare(net_quantized)
    test(net_quantized)

    print(f'Check statistics of the various layers')
    print(net_quantized)

    # 量化？
    net_quantized = torch.ao.quantization.convert(net_quantized)

    print('Weights after quantization')
    print(torch.int_repr(net_quantized.fc1.weight()))

    # Compare the dequantized weights and the original weights
    print('Original weights: ')
    print(net.fc1.weight)
    print('')
    print(f'Dequantized weights: ')
    print(torch.dequantize(net_quantized.fc1.weight()))
    print('')

    torch.save(net_quantized, 'qnet.pt')
    torch.save(net_quantized.state_dict(), 'qnet.pth')

    # Print size and accuracy of the quantized model
    print('Size of the model after quantization')
    print_size_of_model(net_quantized)
    print('Testing the model after quantization')
    # test(net_quantized)

    print('Weights after quantization')
    print(net_quantized.fc1.weight)
    # print(net_quantized.fc1.weight.dtype)

    print('Size of the model after quantization')
    print_size_of_model(net_quantized)
    print(net_quantized)

    print("fc1 weight (quantized):", net_quantized.fc1.weight())
    print("fc1 weight scale:", net_quantized.fc1.weight().q_scale())
    print("fc1 weight zero_point:", net_quantized.fc1.weight().q_zero_point())

    print("fc1 output scale:", net_quantized.fc1.scale)
    print("fc1 output zero_point:", net_quantized.fc1.zero_point)

    print("fc1 bias (float):", net_quantized.fc1.bias())
    test_data = get_data_loader(is_train=False)
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net_quantized.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
