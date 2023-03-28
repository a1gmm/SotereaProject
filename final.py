import cv2
from CAR import CAR
import numpy as np
import time
import torch
from tracks_training import Net

device = torch.device("cpu")
net = Net().to(device)
net.train(False)
net.load_state_dict(torch.load(f"parameters/resnet18_e100.pth", map_location=device))

# 调用模型，输入原始图片，输出运动状态

def auto_drive_with_nn(image: np.ndarray):
    with torch.no_grad():
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)  # 缩放成 224 * 224
        image = np.transpose(image, (2, 0, 1))  # 将RGB通道移到最前面
        image = image.reshape(1, 3, 224, 224)  # 重塑成 batch_size * channel * width * height 形式
        image = torch.tensor(image / 255, dtype=torch.float).to(device)  # 将值从 0-255 映射到 0-1 并转成 tensor
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        moving_status = predicted.item()
        return moving_status
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    car = CAR()

    camera = True
    speed = 40
    frame_mode = 1
    thresh = 110
    roundangle = 160
    upangle = 80
    auto = False
    net = False

    while True:
        key = cv2.waitKey(1)

        car.set_servo_angle(10, roundangle)
        car.set_servo_angle(9, upangle)
        if key == ord("p"):
            upangle += 10
        if key == ord(";"):
            upangle -= 10
        if key == ord("l"):
            roundangle -= 10
        if key == ord("'"):
            roundangle += 10

        turnspeed = speed - 15

        if camera:
            ret, input_frame = capture.read()
        if not ret:
            print("Can't receive frame")
            break

        height, width, _ = input_frame.shape
        gray_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        _, bw_frame = cv2.threshold(gray_frame, thresh, 255, cv2.THRESH_BINARY)
        clean_frame = cv2.dilate(bw_frame, (50, 50), iterations=10)
        _, clean_frame = cv2.threshold(clean_frame, thresh, 255, cv2.THRESH_BINARY)
        clean_frame = cv2.erode(clean_frame, (50, 50), iterations=1)
        clean_frame[:height // 3, :] = 0
        contours, _ = cv2.findContours(clean_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_frame = input_frame.copy()
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_frame, [contour], -1, (0, 0, 255), 10)

        if frame_mode >= 1:
            output_frame = input_frame
        if frame_mode >= 2:
            output_frame = gray_frame
        if frame_mode >= 3:
            output_frame = bw_frame
        if frame_mode >= 4:
            output_frame = clean_frame
        if frame_mode >= 5:
            output_frame = contour_frame

        cv2.line(output_frame, (0, 240), (640, 240), (0, 0, 255), 1, 4)
        cv2.line(output_frame, (320, 0), (320, 480), (0, 0, 255), 1, 4)
        cv2.putText(output_frame, f'speed={speed}', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(output_frame, f'thresh={thresh}', (0, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(output_frame, f'upangle={upangle}', (0, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(output_frame, f'roundangle={roundangle}', (0, 115), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("camera", output_frame)

        if key == ord("z"):
            auto = not auto
        if auto is True:
            car.t_down(speed, 0)
            n_left_bottom_block = np.sum(clean_frame[320:, :320] == 0)
            n_right_bottom_block = np.sum(clean_frame[320:, 320:] == 0)

            if n_left_bottom_block >= 20000:
                car.turn_left(turnspeed, 0)
            elif n_right_bottom_block >= 20000:
                car.turn_right(turnspeed, 0)
            else:
                car.t_down(speed, 0)
            continue

        if key == ord("x"):
            net = not net
        if net is True:
            moving_status= auto_drive_with_nn(input_frame)
            if moving_status == 0:
                car.turn_right(turnspeed, 0)
            elif moving_status == 1:
                car.t_down(speed, 0)
            elif moving_status == 2:
                car.turn_left(turnspeed, 0)
            continue

        if key == ord("j"):
            thresh -= 10
        if key == ord("k"):
            thresh += 10

        if key == ord("q"):
            speed -= 10
        if key == ord("e"):
            speed += 10

        if ord('0') <= key <= ord('9'):
            frame_mode = key - 48

        # 拍照 m
        if key == ord("c"):
            filename = time.strftime("%Y.%m.%d_%H.%M.%S.jpg")
            cv2.imwrite(filename, output_frame)

        if key == ord("s"):
            car.t_up(speed, 0)
        if key == ord("w"):
            car.t_down(speed, 0)
        if key == ord("a"):
            car.turn_right(turnspeed, 0)
        if key == ord("d"):
            car.turn_left(turnspeed, 0)
        if key == ord(" "):
            car.t_stop(0)

        if key == 27:
            car.t_stop(0)
            break

capture.release()
cv2.destroyAllWindows()
