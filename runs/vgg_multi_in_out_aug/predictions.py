import os
import numpy as np
import torch
import pandas as pd
import tqdm
from torchvision.io import read_image
from torchvision.transforms import functional as F
from models.vgg import VGG
from prepare import sin_transformer, cos_transformer

if __name__ == '__main__':
    # load model
    state = torch.load('./best_model.pth')
    model = VGG(3)
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_csv('../../data/test.csv')
    df[df.columns[1:]] = df[df.columns[1:]].apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)), axis=1)
    preds = []

    for f in tqdm.tqdm(os.listdir('../../data/test_images')):
        img = read_image(os.path.join('../../data/test_images', f))
        img = F.resize(img, [128, 128]) / 255.
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(df.loc[int(f.split('.')[0]), :].values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            model.eval()
            y1, y2 = model((img, x2))

        logits = y1 + y2
        preds.append([f.split('.')[0]] + list(logits.numpy()[0]))

    preds = pd.DataFrame(preds, columns=['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
    preds.to_csv('./submission.csv', index=False)
