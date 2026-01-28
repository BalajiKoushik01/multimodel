import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_encoder import DualEncoder
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_swede_scores(csv_path):
    """
    Parses swede_scores.csv.
    Assumes row index i corresponds to 'Case {i+1}'.
    Score < 5: Benign (0)
    Score >= 5: Malignant (1)
    """
    case_labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row: continue
            try:
                score = int(row[0])
                label = 1 if score >= 5 else 0
                case_name = f"Case {i+1}" # e.g., Case 1, Case 2...
                case_labels[case_name] = label
            except ValueError:
                continue
    return case_labels

class EvalDataset(Dataset):
    def __init__(self, root_dir, case_labels, transform=None):
        self.root_dir = root_dir
        self.case_labels = case_labels
        self.transform = transform
        self.items = []

        # Iterate through Case folders
        for case_name, label in case_labels.items():
            case_path = os.path.join(root_dir, case_name)
            if not os.path.exists(case_path):
                continue
            
            # Find images in case folder
            for fname in os.listdir(case_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(case_path, fname)
                    # We treat every image in the case as having that case's label
                    self.items.append((img_path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy
            image = torch.zeros((3, 256, 256))
        
        return image, label

def evaluate():
    config_path = 'configs/config.yaml'
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Data
    swede_path = os.path.join(config['dataset_path'], 'annocerv_real', 'swede_scores.csv')
    if not os.path.exists(swede_path):
        print(f"Error: {swede_path} not found.")
        return

    case_labels = parse_swede_scores(swede_path)
    print(f"Loaded labels for {len(case_labels)} cases.")

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])

    dataset = EvalDataset(os.path.join(config['dataset_path'], 'annocerv_real'), case_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"Evaluating on {len(dataset)} images from AnnoCerv.")

    # Model
    model = DualEncoder(config).to(device)
    
    # Load Weights
    weights_path = os.path.join(config['weights_save_path'], 'finetuned_epoch_100.pth')
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else:
        print("Error: Finetuned weights not found.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            # For inference, if we don't have separate histo images, duplicate colpo
            # or update model to handle single input. 
            # Current DualEncoder expects (img_c, img_h).
            # We will pass same image to both for now as a fallback/proxy
            
            logits_mal, _ = model.forward_downstream(imgs, imgs)
            preds = torch.argmax(logits_mal, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    report = classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'])
    print("\nClassification Report:")
    print(report)

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix (AnnoCerv Evaluation)')
    
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/confusion_matrix.png')
    print("Saved confusion matrix to visualizations/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
