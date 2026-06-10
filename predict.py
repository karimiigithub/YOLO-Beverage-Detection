from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source = "test.mp4" , show = True , save=True)