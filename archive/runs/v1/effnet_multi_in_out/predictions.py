import os

import numpy as np
import torch
import pandas as pd
import tqdm
import albumentations as A
import cv2

from archive.models.approch_1.backbone.effnet import CustomEffNet
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from archive.utils.move import move_to

if __name__ == '__main__':
    TRANSFORMER = Compose([A.Resize(300, 300),
                           A.Normalize(
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225],
                           ),
                           ToTensorV2(),
                           ])

    # load model
    state = torch.load('best_model.pth')
    model = CustomEffNet()
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_csv('../../../../data/test.csv', index_col='id')
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)
    df_sq = df.apply(lambda x: x ** 2, axis=1)
    df_sqrt = df.apply(lambda x: np.sqrt(x), axis=1)
    df = pd.concat((df_sq, df_sqrt, df), axis=1)

    preds = []

    for f in tqdm.tqdm(os.listdir('../../../../data/test_images')):
        img = cv2.imread(os.path.join('../../../../data/test_images', f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = TRANSFORMER(image=img)
        img = augmented['image']
        img = img.unsqueeze(0)
        x2 = torch.tensor(df.loc[int(f.split('.')[0]), :].values, dtype=torch.float32).unsqueeze(0)

        # plt.figure()
        # plt.imshow(augmented['image'].permute(1, 2, 0))
        # plt.show()
        # break

        img = move_to(img, 'cuda')
        x2 = move_to(x2, 'cuda')
        model = move_to(model, 'cuda')

        with torch.no_grad():
            model.eval()
            y1, y2 = model((img, x2))

        logits = torch.abs(y1 + y2 * np.random.normal())
        preds.append([f.split('.')[0]] + list(logits.cpu().numpy()[0]))

    preds = pd.DataFrame(preds, columns=['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
    preds.to_csv('./submission.csv', index=False)
