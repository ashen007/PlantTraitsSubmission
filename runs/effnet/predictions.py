import os

import torch
import pandas as pd
import tqdm
import albumentations as A
import cv2
import joblib

from archive.models.approch_2.effnetlg import CustomEffnetLarge
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from archive.utils.move import move_to

if __name__ == '__main__':
    TRANSFORMER = Compose([A.Resize(256, 256),
                           A.ToFloat(),
                           A.Normalize(
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225],
                           ),
                           ToTensorV2(),
                           ])

    # load model
    state = torch.load('best_checkpoint.pth')
    model = CustomEffnetLarge()
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_csv('../../data/test.csv', index_col='id')
    pipe = joblib.load('../../data/processed/scale.joblib')

    preds = []

    for f in tqdm.tqdm(df.index.tolist()):
        img = cv2.imread(os.path.join('../../data/test_images', f'{f}.jpeg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = TRANSFORMER(image=img)
        img = augmented['image']
        img = img.unsqueeze(0)

        img = move_to(img, 'cuda')
        model = move_to(model, 'cuda')

        with torch.no_grad():
            model.eval()
            y = model(img)

        logits = pipe.inverse_transform(y.cpu().numpy()).squeeze()

        for i in range(1, len(logits)):
            logits[i] = 10 ** logits[i]

        preds.append([f] + list(logits))

    preds = pd.DataFrame(preds, columns=['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112']).set_index('id')

    # restore to original scale
    preds.to_csv('./submission.csv', index=True)
