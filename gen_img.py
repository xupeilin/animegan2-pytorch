import argparse

import torch
import cv2
import numpy as np
import os
from datetime import datetime as dt

from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
def load_image(image_path, x32=False, zoom_pct=100):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print("Image original shape ", img.shape)

    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w/2), to_32s(h/2)))
    img = cv2.resize(img, (int(w*zoom_pct/100), int(h*zoom_pct/100)))
    print("Image shape after resize ", img.shape)

    img = torch.from_numpy(img)
    img = img/127.5 - 1.0
    return img


def generate(args):
    device = args.device
    
    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {args.checkpoint}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    image_name = args.input
    if os.path.splitext(image_name)[-1].lower() not in [".jpeg", ".jpg", ".png", ".bmp", ".tiff"]:
        print("Input image error ", image_name)
        exit
        
    image = load_image(image_name, args.x32, 100)

    with torch.no_grad():
        input = image.permute(2, 0, 1).unsqueeze(0).to(device)
        out = net(input, args.upsample_align).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out + 1)*127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        
    img_field = image_name.split(".")
    output_image_name = img_field[0]+"-gen-" + dt.now().strftime("%y%m%d-%H%M%S") + "." + img_field[1]
    cv2.imwrite(output_image_name, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    print(f"image saved: {output_image_name}")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        #default='./pytorch_generator_Paprika.pt',
        default='./pytorch_generator_Hayao.pt',
        #default='./pytorch_generator_Shinkai.pt',
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='./samples.jpg',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--x32',
        action="store_true",
    )
    args = parser.parse_args()
    
    generate(args)
    
