from net_quan import *


def save_quantized_weights_verilog(model, output_dir="dw_quant_weights_verilog"):
    os.makedirs(output_dir, exist_ok=True)
    weight_file = os.path.join(output_dir, f"quant_weight.txt")
    with open(weight_file, "w") as f:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.quantized.Conv2d):
                # 获取权重整数表示
                weight_tensor = module.weight().int_repr().cpu().numpy()
                shape = weight_tensor.shape  # (out_channels, in_channels, kernel_h, kernel_w)

                # f.write(f"\n---------------Quantize Layer {name}---------------\n")
                # # 写入 scale 和 zero_point
                # f.write(f"scale: {module.scale}\n")
                # f.write(f"zero_point: {module.zero_point}\n")
                # # 写入权重
                # f.write(f"# Shape: {shape}\n")
                # for oc in range(shape[0]):
                #     for ic in range(shape[1]):
                #         for kh in range(shape[2]):
                #             for kw in range(shape[3]):
                #                 val = weight_tensor[oc][ic][kh][kw]
                #                 f.write(f"{val} ")
                #     f.write(f"\n")
            else:
                if isinstance(module, torch.nn.quantized.modules.Quantize):
                    pass
                    # f.write(f"\n---------------Quantize Layer {name}---------------\n")
                    # f.write(f"  Scale: {module.scale.item()}\n")
                    # f.write(f"  Zero Point: {module.zero_point.item()}\n")
                    # f.write(f"  DType: {module.dtype}\n\n")
    print(f"所有参数已保存到：{weight_file}")


def main():
    qnet = QuanNet().to(device)
    qnet.eval()
    qnet.qconfig = torch.ao.quantization.default_qconfig
    qnet = torch.ao.quantization.prepare(qnet)
    qnet = torch.ao.quantization.convert(qnet)
    qnet.load_state_dict(torch.load('./qnet.pth'))
    save_quantized_weights_txt(qnet)


if __name__ == "__main__":
    main()
