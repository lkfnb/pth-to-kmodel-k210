import torch

from model import K210_YOLO


INPUT_SIZE = 224
NUM_CLASSES = 3
NUM_ANCHORS = 5
WEIGHT_PATH = "./weights/best.pth"
ONNX_PATH = "yolo_k210.onnx"

device = torch.device("cpu")


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_raw(x)


model = K210_YOLO(
    num_classes=NUM_CLASSES,
    num_anchors=NUM_ANCHORS
)

model.load_state_dict(
    torch.load(
        WEIGHT_PATH,
        map_location=device
    )
)

model.eval()

export_model = ExportWrapper(model)
export_model.eval()

print("model loaded")

dummy_input = torch.randn(
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE
)

with torch.no_grad():
    y = export_model(dummy_input)
    print("export output shape:", y.shape)

torch.onnx.export(
    export_model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes=None
)

print("onnx exported:", ONNX_PATH)