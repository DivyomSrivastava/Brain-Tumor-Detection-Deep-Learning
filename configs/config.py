IMAGE_SIZE = 224

NUM_CLASSES = 4

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

NUM_EPOCHS = 25

DROPOUT = 0.3

PATIENCE = 5

TRAIN_DIR = "dataset/brain_mri/Training"

TEST_DIR = "dataset/brain_mri/Testing"

MODEL_PATH = "models/best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"