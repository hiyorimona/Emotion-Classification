import pandas as pd
import os
import torch
from tqdm import tqdm
from src.bert import bert_model
from src.bert.bert_tokenizer import TokenizedDataset
from torch.utils.data import DataLoader
from collections import Counter
from nltk.tokenize import sent_tokenize
from preprocessing import processor
from transformers import BertTokenizer


class DataPredictor:

    def tokenize_sentences(self, text):
        return sent_tokenize(text)

    def split_texts(self, df_ep):
        indexes = []
        list_sentences = df_ep['text'].apply(lambda x: self.tokenize_sentences(x)).tolist()
        sentences = []
        for idx, text in enumerate(list_sentences):
            sentences.extend(text)
            for _ in text:
                indexes.append(idx)
        df = pd.DataFrame({'id_chunk': indexes,
                           'sentence': sentences})
        return df

    def find_top3_dominating_emotions(self, emotion_string):
            emotions = emotion_string.split(', ')
            emotion_counts = Counter(emotions)
            top3_emotions = emotion_counts.most_common(3)

            return [emotion for emotion, count in top3_emotions]


    def group_and_combine(self, df_transcription):
        grouped_df_sentences = df_transcription.groupby('id_chunk')['sentence'].apply(lambda x: ' '.join(x)).reset_index()
        grouped_df_emotions = df_transcription.groupby('id_chunk')['emotion'].apply(lambda x: ', '.join(x)).reset_index()

        combined_df = pd.concat([grouped_df_sentences['sentence'], grouped_df_emotions['emotion']], axis=1)
        combined_df.rename(columns={'sentence': 'segment', 'emotion': 'emotions'}, inplace=True)
        combined_df['dominating_emotions'] = combined_df['emotions'].apply(self.find_top3_dominating_emotions)

        return combined_df


    def dataset_loader(self, df_transcription,tokenizer, batch_size, max_len):
        df_sentences = self.split_texts(df_transcription)
        tokenized_dataset = TokenizedDataset(df_sentences, tokenizer, max_len)
        data_loader = torch.utils.data.DataLoader(tokenized_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

        return data_loader, df_sentences


    def get_finetuned_model(self, model_path,device):
        if model_path.endswith('.bin'):
            current_dir = os.getcwd()
            parent_dir_components = current_dir.split(os.path.sep)[:2]
            parent_dir_components.extend(['BERT_classification', 'models', model_path])

            model_dir = os.path.sep.join(parent_dir_components)
            state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
            model = bert_model.BERTClassification()
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            return model
        else:
            print('missing .bin extension')


    def get_predictions(self, df_transcription, transformer_model, tokenizer, batch_size, max_len, device):

        data_loader, df_sentences = self.dataset_loader(df_transcription,tokenizer,batch_size,max_len)
        emotion_mapping = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5}

        transformer_model.eval()
        predictions = []

        with torch.no_grad():
            for data in data_loader:
                ids = data["input_ids"].to(device)
                mask = data["attention_mask"].to(device)
                token_type_ids = data['token_type_ids'].to(device)

                outputs = transformer_model(ids, mask, token_type_ids)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.tolist())

        predicted_emotions = [emotion for label in predictions for emotion, encoded_label in emotion_mapping.items() if label == encoded_label]

        df_sentences['emotion'] = predicted_emotions
        df_sentences = self.group_and_combine(df_sentences)

        return df_sentences

    def run(self, df_transcription, path, tokenizer,transformer_model, batch_size, max_len, device):
        files = sorted(os.listdir(path), key=processor.extract_number)
        for file in tqdm(files):

            full_path = f"{path}/{file}"
            df_predictions = self.get_predictions(df_transcription, transformer_model, tokenizer, batch_size, max_len, device)
            df_predictions.to_csv(f"{full_path}/labeled_transcription.csv", index=False)


if __name__ == "__main__":
    predictor = DataPredictor()
    MAX_LEN = 256
    BATCH_SIZE = 32

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = predictor.get_finetuned_model('best_model_state.bin', device)


