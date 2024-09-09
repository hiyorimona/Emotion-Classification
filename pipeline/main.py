path = "../episodes"
model = whisper.load_model("large")

structure_df = pd.read_csv(f"/Robinson22_structure.csv")
files = os.listdir(path)
model = whisper.load_model("large")
path = "../Data"
structure_df = pd.read_csv(f"{path}/Robinson22_structure.csv")

MAX_LEN = 256
BATCH_SIZE = 32

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')