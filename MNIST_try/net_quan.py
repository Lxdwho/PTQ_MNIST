import torch
from MNIST_CNN import get_data_loader, evaluate, device
import os
from net_detect import detect_fun


def save_quantized_weights_txt(model, output_dir="dw_quant_weights_txt"):
    os.makedirs(output_dir, exist_ok=True)
    weight_file = os.path.join(output_dir, f"quant_weight.txt")
    with open(weight_file, "w") as f:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.quantized.Conv2d):
                # 获取权重整数表示
                weight_tensor = module.weight().int_repr().cpu().numpy()
                shape = weight_tensor.shape  # (out_channels, in_channels, kernel_h, kernel_w)

                f.write(f"\n---------------Quantize Layer {name}---------------\n")
                # 写入 scale 和 zero_point
                f.write(f"scale: {module.scale}\n")
                f.write(f"zero_point: {module.zero_point}\n")
                # 写入权重
                f.write(f"# Shape: {shape}\n")
                for oc in range(shape[0]):
                    for ic in range(shape[1]):
                        for kh in range(shape[2]):
                            for kw in range(shape[3]):
                                val = weight_tensor[oc][ic][kh][kw]
                                f.write(f"{val} ")
                    f.write(f"\n")
            else:
                if isinstance(module, torch.nn.quantized.modules.Quantize):
                    f.write(f"\n---------------Quantize Layer {name}---------------\n")
                    f.write(f"  Scale: {module.scale.item()}\n")
                    f.write(f"  Zero Point: {module.zero_point.item()}\n")
                    f.write(f"  DType: {module.dtype}\n\n")
    print(f"所有参数已保存到：{weight_file}")


class QuanNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        # self.cn1 = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=1)# 28 * 28 * 3   1 * 3 * 3 * 3 = 27
        # self.mp1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=3, padding=0, bias=False, groups=1)# 09 * 09 * 3   3 * 3 * 3 * 3 = 81
        # self.cn2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False, groups=1)# 09 * 09 * 6   3 * 3 * 3 * 6 = 162
        # self.mp2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=3, padding=0, bias=False, groups=1)# 03 * 03 * 6   6 * 3 * 3 * 6 = 324
        # self.cn3 = torch.nn.Conv2d(6, 10, kernel_size=3, stride=1, padding=1, bias=False, groups=1)# 03 * 03 * 10  6 * 3 * 3 * 10 = 540
        # self.mp3 = torch.nn.Conv2d(10, 10, kernel_size=3, stride=3, padding=0, bias=False, groups=1)# 01 * 01 * 10  10 * 3 * 3 * 10 = 900
        self.dw1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=3, padding=0, bias=False, groups=1)  # 09 * 09 * 1
        self.pw1 = torch.nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False, groups=1)  # 09 * 09 * 3
        self.dw2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=3, padding=0, bias=False, groups=3)  # 03 * 03 * 3
        self.pw2 = torch.nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0, bias=False, groups=1)  # 03 * 03 * 6
        self.dw3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=3, padding=0, bias=False, groups=6)  # 03 * 03 * 6
        self.pw3 = torch.nn.Conv2d(6, 10, kernel_size=1, stride=1, padding=0, bias=False, groups=1)  # 01 * 01 * 10
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        # x = torch.nn.functional.relu(self.cn1(x))
        # x = torch.nn.functional.relu(self.mp1(x))
        # x = torch.nn.functional.relu(self.cn2(x))
        # x = torch.nn.functional.relu(self.mp2(x))
        # x = torch.nn.functional.relu(self.cn3(x))
        # x = self.mp3(x)
        # x = self.dequant(x)
        # x = torch.nn.functional.log_softmax(x.view(-1, 10), dim=1)
        x = self.quant(x)
        x = torch.nn.functional.relu(self.pw1(self.dw1(x)))
        x = torch.nn.functional.relu(self.pw2(self.dw2(x)))
        x = torch.nn.functional.relu(self.pw3(self.dw3(x)))
        x = self.dequant(x)
        x = torch.nn.functional.log_softmax(x.view(-1, 10), dim=1)

        return x


def main():
    qnet = QuanNet().to(device)
    qnet.load_state_dict(torch.load('./weights/20net.pth'))

    qnet.qconfig = torch.ao.quantization.default_qconfig
    qnet = torch.ao.quantization.prepare(qnet)

    qnet.eval()
    test_data = get_data_loader(is_train=False)
    print("Accuracy:", evaluate(test_data, qnet))

    qnet = torch.ao.quantization.convert(qnet)
    print("Accuracy:", evaluate(test_data, qnet))
    detect_fun(qnet, test_data)

    torch.save(qnet, 'qnet.pt')
    torch.save(qnet.state_dict(), 'qnet.pth')
    save_quantized_weights_txt(qnet)

    print(qnet)


if __name__ == "__main__":
    main()
