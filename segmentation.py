import numpy as np
import torch
from PIL import Image

def segmentation_output(mask, num_classes=7):
    label_colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0),(256, 256, 256), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # 0: 전신, 1: 머리카락, 2: 머리~목, 3: 상의, 4: 바지, 5: 배경, 6: 팔

    h, w = mask.shape
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    output = np.array(img)

    return output


def segementation(image, model):
    test_conv = image.transpose(2, 0 ,1)
    test_conv1 = test_conv[np.newaxis, :, :, :]
    test_conv_tensor = torch.from_numpy(test_conv1.copy()).float()
    conv_out_model = model(test_conv_tensor)
    output_model = torch.argmax(conv_out_model, dim=1)
    vis_output_model = segmentation_output(output_model[0].data.cpu().numpy())
    return vis_output_model