from ultralytics import YOLO

# load a model
model = YOLO("yolov8s.pt") #build a new model from scratch

# use the model
results = model.train(data="config.yaml", epochs=3, batch=2) # train the model