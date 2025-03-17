import json
from pathlib import Path
import numpy as np
from PIL import Image
from dataset import config


class TextVQAX:
    def __init__(self):
        # train data range [width, height] = [256, 256] [6000, 4000]
        # average data range [width, height] = [1024, 1024]
        self.base_path = Path(config.TEXT_VQA_X_PATH)
        self.text_vqa_path = Path(config.TEXT_VQA_PATH)
        self.n_training = 14476
        self.n_validation = 3620
        self.n_classes = 2
        self.class_colors = [[25, 25, 25], [230, 230, 230]]

        with open(self.base_path / 'text_explanations' / 'explanations_token.json') as file:
            self.text_explanations = json.load(file)

        self.text_vqa_data = {}
        for dataset in ['train', 'val', 'test']:
            with open(self.text_vqa_path / f'TextVQA_0.5.1_{dataset}.json') as file:
                data = json.load(file)['data']
                for q in data:
                    self.text_vqa_data[str(q['question_id'])] = q

    def load_training_list(self):
        return self._load_files_list('train')

    def load_validation_list(self):
        return self._load_files_list('val')

    def _load_files_list(self, dataset):
        with open(self.base_path / f'data_splits/{dataset}_id.txt') as file:
            ids = file.read().splitlines()

        images = []
        labels = []
        explanations = []
        questions = []
        for key in self.text_explanations.keys():
            if key in ids:
                images.append('train_images/' + self.text_explanations[key]['image'])
                labels.append(f'visual_explanations/seg/{key}.npy')
                explanations.append(self.text_explanations[key]['explanation'])
                questions.append(self.text_vqa_data[key]['question'])
        to_array = lambda x: np.array(x, dtype=object)
        return to_array(images), to_array(labels), to_array(explanations), to_array(questions)

    def load_files_pil(self, image_file, label_file):
        image = Image.open(str(self.text_vqa_path / image_file)).convert('RGB')
        label = np.load(self.base_path / label_file)
        label = Image.fromarray(np.uint8(label))
        return image, label

    def load_files(self, image_file, label_file):
        image = Image.open(str(self.text_vqa_path / image_file)).convert('RGB')
        label = np.load(self.base_path / label_file)
        image = np.array(image)
        label = np.uint8(label)
        return image, label
