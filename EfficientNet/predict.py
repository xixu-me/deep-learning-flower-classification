import json
import os

import matplotlib.pyplot as plt
import torch
from model import efficientnet_b0 as create_model
from PIL import Image
from torchvision import transforms


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {
        "B0": 224,
        "B1": 240,
        "B2": 260,
        "B3": 300,
        "B4": 380,
        "B5": 456,
        "B6": 528,
        "B7": 600,
    }
    num_model = "B0"

    data_transform = transforms.Compose(
        [
            transforms.Resize(img_size[num_model]),
            transforms.CenterCrop(img_size[num_model]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_path = "../dress.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = img.convert("RGB")
    plt.imshow(img)

    img = data_transform(img)

    img = torch.unsqueeze(img, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = create_model(num_classes=5).to(device)
    model_weight_path = "./weights/model-1.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy()
    )
    plt.title(print_res)
    for i in range(len(predict)):
        print(
            "class: {:10}   prob: {:.3}".format(
                class_indict[str(i)], predict[i].numpy()
            )
        )
    plt.show()


if __name__ == "__main__":
    main()
