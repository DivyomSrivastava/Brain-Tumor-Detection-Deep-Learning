"""
------------------------------------------------------------
NeuroVision AI

Brain MRI Tumor Prediction Module

Author : Divyom Srivastava
Framework : PyTorch
------------------------------------------------------------
"""


# ============================================================
# Imports
# ============================================================

import torch

from PIL import Image


from configs.config import DEVICE

from src.models.efficientnet import BrainTumorClassifier

from src.data.transforms import test_transform

from src.data.dataloader import CLASS_NAMES



# ============================================================
# Model Loading
# ============================================================


MODEL_PATH = "models/best_model.pth"


device = torch.device(DEVICE)



model = BrainTumorClassifier(
    freeze_features=False
).to(device)



model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device
    )

)


model.eval()



# ============================================================
# Prediction Function
# ============================================================


def predict_image(image_path):

    """
    Predict brain tumor class from MRI image


    Parameters
    ----------
    image_path : str

        Path of MRI image


    Returns
    -------

    prediction : str

        Predicted tumor class


    confidence : float

        Prediction confidence percentage


    probabilities : dict

        Probability of every class

    """


    # --------------------------------------------------------
    # Load Image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")



    # --------------------------------------------------------
    # Transform Image
    # --------------------------------------------------------

    input_tensor = test_transform(
        image
    )


    input_tensor = input_tensor.unsqueeze(
        0
    ).to(device)



    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )


        probabilities_tensor = torch.softmax(
            output,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities_tensor,
            1
        )



    # --------------------------------------------------------
    # Convert Output
    # --------------------------------------------------------


    prediction = CLASS_NAMES[
        predicted.item()
    ]



    confidence = (
        confidence.item()
        *
        100
    )



    probability_values = (
        probabilities_tensor
        .squeeze()
        .cpu()
        .numpy()
        *
        100
    )



    probabilities = {

        CLASS_NAMES[i]:
        round(
            float(probability_values[i]),
            2
        )

        for i in range(
            len(CLASS_NAMES)
        )

    }



    return (

        prediction,

        round(
            confidence,
            2
        ),

        probabilities

    )



# ============================================================
# Terminal Testing
# ============================================================


def print_prediction(image_path):


    prediction, confidence, probabilities = predict_image(
        image_path
    )



    print("\n================================")
    print("NeuroVision AI Prediction")
    print("================================")


    print(
        f"Tumor Type : {prediction}"
    )


    print(
        f"Confidence : {confidence}%"
    )


    print("\nClass Probabilities")
    print("--------------------------------")



    for key,value in probabilities.items():

        print(
            f"{key:15s}: {value}%"
        )


    print("================================")



# ============================================================
# Main Testing
# ============================================================


if __name__ == "__main__":


    IMAGE_PATH = input(
        "Enter MRI image path : "
    ).strip()



    print_prediction(
        IMAGE_PATH
    )