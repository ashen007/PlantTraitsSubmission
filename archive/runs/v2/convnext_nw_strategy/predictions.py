import os

import numpy as np
import torch
import pandas as pd
import tqdm
import albumentations as A
import cv2
import joblib

from archive.models.approch_1.backbone.v2 import CustomConvNextSmall
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from archive.utils.move import move_to

if __name__ == '__main__':
    TRANSFORMER = Compose([A.Resize(256, 256),
                           A.Normalize(
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225],
                           ),
                           ToTensorV2(),
                           ])

    # load model
    state = torch.load('best_model.pth')
    model = CustomConvNextSmall()
    model.load_state_dict(state['model_state_dict'])

    df = pd.read_csv('../../../../data/test.csv', index_col='id')

    scaler = joblib.load('../../../data/scaler.joblib')
    pipe = joblib.load('../../../data/pipe.joblib')

    # train data scaler
    df[df.columns] = scaler.transform(np.asarray(df))

    preds = []

    for f in tqdm.tqdm(os.listdir('../../../../data/test_images')):
        img = cv2.imread(os.path.join('../../../../data/test_images', f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = TRANSFORMER(image=img)
        img = augmented['image']
        img = img.unsqueeze(0)
        x2 = torch.tensor(df.loc[int(f.split('.')[0]), :].values, dtype=torch.float32).unsqueeze(0)

        img = move_to(img, 'cuda')
        x2 = move_to(x2, 'cuda')
        model = move_to(model, 'cuda')

        with torch.no_grad():
            model.eval()
            y = model((img, x2))

        logits = y
        preds.append([f.split('.')[0]] + list(logits.cpu().numpy()[0]))

    preds = pd.DataFrame(preds, columns=['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112']).set_index('id')
    preds[preds.columns] = np.asarray(pipe.inverse_transform(np.asarray(preds)))

    # restore to original scale
    preds.to_csv('./submission_1.csv', index=True)
