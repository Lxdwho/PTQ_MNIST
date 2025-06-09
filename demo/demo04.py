import os
from demo02 import *
from demo03 import *


def save_model_parameters_txt(model, folder="qmodel_params_txt"):
    os.makedirs(folder, exist_ok=True)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 保存权重
            weight = module.weight.detach().cpu().numpy()
            weight_file = os.path.join(folder, f"{name}_weight.txt")
            with open(weight_file, "w") as f:
                for row in weight:
                    f.write(" ".join([f"{w:.6f}" for w in row]) + "\n")

            # 保存偏置
            bias = module.bias.detach().cpu().numpy()
            bias_file = os.path.join(folder, f"{name}_bias.txt")
            with open(bias_file, "w") as f:
                f.write(" ".join([f"{b:.6f}" for b in bias]) + "\n")

            print(f"✅ Saved: {name} -> weight & bias")


def main():
    net = QuantizedVeryNet()
    net.load_state_dict(torch.load('qnet.pth'))
    save_model_parameters_txt(net)


if __name__ == "__main__":
    main()
