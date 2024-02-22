import os
import numpy as np
import torch
import pandas as pd
import tqdm
from torchvision.io import read_image
from archive.models.approch_1.regnet import RegNet
from torchvision.transforms.v2 import (Compose,
                                       Normalize,
                                       ToDtype,
                                       Resize)
from archive.utils.move import move_to

if __name__ == '__main__':
    TRANSFORMER = Compose([Resize(128),
                           ToDtype(torch.float32),
                           Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ])

    # load model
    state = torch.load('best_model.pth')
    model = RegNet(3)
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_csv('../../../../data/test.csv', index_col='id')
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)
    df_sq = df.apply(lambda x: x ** 2, axis=1)
    df_sqrt = df.apply(lambda x: np.sqrt(x), axis=1)
    df = pd.concat((df_sq, df_sqrt, df), axis=1)

    preds = []

    for f in tqdm.tqdm(os.listdir('../../../../data/test_images')):
        img = read_image(os.path.join('../../../../data/test_images', f)) / 255.
        img = TRANSFORMER(img).unsqueeze(0)
        x2 = torch.tensor(df.loc[int(f.split('.')[0]), :].values, dtype=torch.float32).unsqueeze(0)

        img = move_to(img, 'cuda')
        x2 = move_to(x2, 'cuda')
        model = move_to(model, 'cuda')

        with torch.no_grad():
            model.eval()
            y1, y2 = model((img, x2))

        logits = torch.abs(y1 + y2)
        preds.append([f.split('.')[0]] + list(logits.cpu().numpy()[0]))

    preds = pd.DataFrame(preds, columns=['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
    preds.to_csv('./submission_0.csv', index=False)
