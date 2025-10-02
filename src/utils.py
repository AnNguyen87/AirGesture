"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import tensorflow as tf
from time import sleep
from src.config import HAND_GESTURES


def is_in_triangle(point, triangle):
    # barycentric coordinate system
    x, y = point
    (xa, ya), (xb, yb), (xc, yc) = triangle
    a = ((yb - yc)*(x - xc) + (xc - xb)*(y - yc)) / ((yb - yc)*(xa - xc) + (xc - xb)*(ya - yc))
    b = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / ((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
    c = 1 - a - b
    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
        return True
    else:
        return False


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    
    sess = tf.compat.v1.Session(graph=graph)
    return graph, sess


def detect_hands(image, graph, sess):
    input_array = np.array(image, dtype=np.uint8)
    input_array = np.expand_dims(input_array, axis=0)

    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    
    (boxes, scores, classes) = sess.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: input_array})
    
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    count = 0
    results = {}
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results[count] = [x_min, x_max, y_min, y_max, category]
            count += 1
    return results

def dinosaur(v, lock):
    try:
        print("🦖 Opening Dino Game in browser...")
        import webbrowser
        import time
        import pyautogui
        
        webbrowser.open('https://elgoog.im/t-rex/')
        time.sleep(5)
        
        print("Browser opened! Game should be loading...")
        print("If needed, click on the game and press SPACE to start")
        
        # Try to focus and start the game
        try:
            pyautogui.click(500, 400)
            time.sleep(1)
            pyautogui.press('space')
            print("Game started! Controlling with hand gestures...")
        except:
            print("Please start the game manually (click + SPACE)")
        
        print("Hand controls:")
        print("Closed hand: Run")
        print("Open hand in TOP half: JUMP (Space)")
        print("Open hand in BOTTOM half: DUCK (Down arrow)")
        
        last_action = None
        while True:
            with lock:
                action = v.value
            
            # Only send key presses when action changes
            if action != last_action:
                if action == 1:  # Jump
                    pyautogui.press('space')
                    print("JUMP!")
                elif action == 2:  # Duck
                    pyautogui.keyDown('down')
                    print("⬇DUCKING...")
                elif last_action == 2:  # Was ducking, now stop
                    pyautogui.keyUp('down')
                    print("STANDING UP")
                
                last_action = action
            
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Game error: {e}")
        print("Running in simulation mode...")
        import time
        action_names = {0: "RUN", 1: "JUMP", 2: "DUCK"}
        last_action = None
        while True:
            with lock:
                action = v.value
            if action != last_action:
                print(f"{action_names.get(action, 'RUN')}")
                last_action = action
            time.sleep(0.1)