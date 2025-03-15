# import libraries
from fastai.vision.all import *

# เรียก Dataset จากโฟลเดอร์
dataset_train_dir = "../Neural Network/test"  # โฟลเดอร์ train
dataset_valid_dir = "../Neural Network/valid"  # โฟลเดอร์ valid

def get_dls(train_path, valid_path, img_size=224, batch_size=32):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=Resize(img_size),
        batch_tfms=aug_transforms()
    )
    return dblock.dataloaders(Path(train_path).parent, bs=batch_size)

# โหลดข้อมูล
dls = get_dls(dataset_train_dir, dataset_valid_dir)

# โหลด Pre-trained Model (ResNet-34)
learn = cnn_learner(dls, resnet34, metrics=accuracy)

# เทรน model
learn.fine_tune(10)

# Save model
learn.export("dice_cnn_model.pkl")

print("Model training complete. Saved as dice_cnn_model.pkl")