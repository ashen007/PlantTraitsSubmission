import os

import numpy as np
import torch
import pandas as pd
import tqdm
import albumentations as A
import cv2
import joblib

from archive.models.approch_2.effnetlg import CustomConvNextSmall
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from archive.utils.move import move_to

if __name__ == '__main__':
    TRANSFORMER = Compose([A.Resize(128, 128),
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
    pipe = joblib.load('../../../data/target_pipe.joblib')

    preds = []

    for f in tqdm.tqdm(os.listdir('../../../../data/test_images')):
        img = cv2.imread(os.path.join('../../../../data/test_images', f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = TRANSFORMER(image=img)
        img = augmented['image']
        img = img.unsqueeze(0)

        img = move_to(img, 'cuda')
        model = move_to(model, 'cuda')

        with torch.no_grad():
            model.eval()
            y = model(img)

        logits = y
        preds.append([f.split('.')[0]] + list(logits.cpu().numpy()[0]))

    preds = pd.DataFrame(preds, columns=['id', 'X4', 'X11', 'X18', 'X50', 'X26', 'X3112']).set_index('id')
    preds[preds.columns] = pipe.inverse_transform(np.asarray(preds))

    # restore to original scale
    preds.to_csv('./submission.csv', index=True)
