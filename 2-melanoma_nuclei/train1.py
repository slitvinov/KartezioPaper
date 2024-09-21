import argparse

from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.endpoint import (
    EndpointEllipse,
    EndpointHoughCircle,
    EndpointLabels,
    EndpointWatershed,
    LocalMaxWatershed,
)
from kartezio.preprocessing import TransformToHED, TransformToHSV
from kartezio.training import train_model
from numena.io.drive import Directory

model = create_instance_segmentation_model(
    generations=10,
    _lambda=5,
    inputs=3,
    outputs=2,
    endpoint=EndpointWatershed(),
)
elite, _ = train_model(model, read_dataset("dataset"), ".")
