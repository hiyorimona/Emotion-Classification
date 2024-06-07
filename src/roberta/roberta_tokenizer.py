import torch

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, emotions_list):
        """
        Parameters:
            df (pandas.DataFrame): The DataFrame containing column sentence.
            tokenizer: RoBERTa tokenizer used to encode sentences.
            max_len (int): The maximum length of the input sequences.
            emotions_list (list): List of target columns in the DataFrame.
        """
        self.tokenizer = tokenizer
        self.df = df
        self.sentences = list(df['sentence'])  
        self.emotions = self.df[emotions_list].values  
        self.max_len = max_len

    def __len__(self):
        """
        Returns:
            int: The length of the dataset.
        """
        return len(self.sentences)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the given index.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, token_type_ids, targets, and sentence.
        """
        sentence = str(self.sentences[index])  # Get the sentence at the given index
        sentence = " ".join(sentence.split())  # Tokenize the sentence
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'emotion': torch.FloatTensor(self.emotions[index]),
            'sentence': sentence 
        }